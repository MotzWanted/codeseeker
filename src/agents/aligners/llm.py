import abc
import ast
import json
import re
import typing as typ

from jinja2 import Environment, FileSystemLoader
from loguru import logger
import numpy as np
from prompt_poet import Prompt
import pydantic

from agents.base import PATH_TO_TEMPLATES, Aligner
from agents.models import Alignment

from throughster import ModelInterface
from throughster.core.models import BaseResponse, LogProbs, ResponseChoice
from throughster.core.errors import StructuredResponseError


class ReasoningModel(pydantic.BaseModel):
    """Class representing the document indices that substantiates a given fact."""

    chain_of_thought: str = pydantic.Field(..., description="The chain of thought.", max_length=1000)
    code_ids: typ.List[int] = pydantic.Field(
        ...,
        description="A list of ids that support the hypothesis. If no documents supports the hypothesis, return a zero list.",  # noqa: E501
        max_length=20,
    )

    @pydantic.field_validator("code_ids", mode="before")
    @classmethod
    def validate_indices(cls: type["ReasoningModel"], v: typ.List[int]) -> typ.List[int]:
        """Validate labels."""
        if len(v) == 0 or (len(v) > 1 and 0 in v):
            v = [0]
        v = list(set(v))
        return v


def custom_tojson(value):
    # Use json.dumps with ensure_ascii=False to avoid unnecessary escaping
    def sanitize_value(val):
        # Recursively sanitize strings within nested structures
        if isinstance(val, str):
            # Replace non-printable characters with a space
            return re.sub(r"[^\x20-\x7E]", " ", val)
        return val

    sanitized_value = sanitize_value(value)
    return json.dumps(sanitized_value, ensure_ascii=False)


class LLMAligner(Aligner):
    """A LLM-based alignment model"""

    def __init__(
        self, prompt_name: str, num_shots: int, token_limit: int, seed: int, sampling_params: dict[str, typ.Any]
    ):
        env = Environment(loader=FileSystemLoader(PATH_TO_TEMPLATES))
        loader = typ.cast(FileSystemLoader, env.loader)
        self.raw_template, self.template_path, _ = loader.get_source(env, f"{prompt_name}.yml.j2")
        self.prompt_name = prompt_name
        self.num_shots = num_shots
        self.token_limit = token_limit
        self.seed = seed
        self.sampling_params = sampling_params

    async def predict(self, client: ModelInterface, classes: list[dict[str, typ.Any]], **kwargs) -> Alignment:
        """Align a list of segment elements to a class."""

        sparse_vector, probs_vector, reasoning = await self.generate_alignments(
            client=client, classes=classes, **kwargs
        )
        indexes = (np.where(sparse_vector == 1)[0] + 1).tolist()

        return Alignment(indexes=indexes, matrix=sparse_vector, probabilities=probs_vector, response=reasoning)

    @abc.abstractmethod
    async def generate_alignments(
        self, client: ModelInterface, classes: list[str], row: dict[str, typ.Any], **kwargs
    ) -> list[tuple[np.ndarray, np.ndarray]]: ...

    @staticmethod
    def normalized_exp_values(top_logprobs: np.ndarray, axis: int) -> np.ndarray:
        """Calculate the normalized probabilities from the log probabilities."""
        # Subtract the maximum log probability for numerical stability (log-sum-exp)
        max_logprobs = np.max(top_logprobs, axis=axis, keepdims=True)

        # Calculate the unnormalized probabilities using the stabilized log values
        unnormalized_probs = np.exp(top_logprobs - max_logprobs)

        sum_probs = np.sum(unnormalized_probs, axis=axis, keepdims=True)
        normalized_probs = unnormalized_probs / sum_probs
        return np.round(normalized_probs, 4)

    def format_prompt(self, classes: list[dict[str, typ.Any]], **kwargs) -> Prompt:
        """Format the prompt."""
        return Prompt(
            raw_template=self.raw_template,
            template_data={"classes": classes, "custom_tojson": custom_tojson, **kwargs},
            token_limit=self.token_limit,
            # truncation_step=truncation_step,
        )

    @staticmethod
    def prompt_messages_or_string(client: ModelInterface, prompt: Prompt) -> str | list[dict[str, str]]:
        if client.endpoint == "chat/completions":
            return prompt.messages
        return prompt.string

    def parse_response(self, preds: list[int], probs: list[float], classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Parse the response and return the alignment."""
        sparse_vector = np.zeros(len(classes))
        probs_vector = np.zeros(len(classes))
        for i, _ in enumerate(classes, start=1):
            if i not in preds:
                continue
            sparse_vector[i - 1] = 1
            probs_vector[i - 1] = probs[preds.index(i)]
        return sparse_vector, probs_vector

    def _lookup_logprobs(self, logprobs: LogProbs, classes_size: int) -> LogProbs:
        """Based on a list of tokens we need to find logprobs of those tokens that represent the alignment predictions.
        We are looking for numeric tokens and the token "[]" which represents the absence of an alignment ("[0]").
        """
        _logprobs = logprobs.model_copy()
        valid_indices = [
            i for i, token in enumerate(logprobs.tokens) if self._is_predicted_alignment(token, classes_size)
        ]

        _logprobs.text_offset = [logprobs.text_offset[i] for i in valid_indices]
        _logprobs.token_logprobs = [logprobs.token_logprobs[i] for i in valid_indices]
        _logprobs.tokens = [logprobs.tokens[i] for i in valid_indices]
        _logprobs.top_logprobs = [
            self._lookup_top_logprobs(logprobs.top_logprobs[i], classes_size) for i in valid_indices
        ]

        return _logprobs

    def _lookup_top_logprobs(self, top_logprobs: dict[str, float], classes_size: int) -> dict[str, float]:
        return {k: v for k, v in top_logprobs.items() if self._is_predicted_alignment(k, classes_size)}

    @staticmethod
    def _is_predicted_alignment(token: str, classes_size: int) -> bool:
        return (token.isdigit() and int(token) < classes_size) or "[]" in token


class UnconstrainedLLMAligner(LLMAligner):
    """A LLM-based Alignment model applying long-context retrieval."""

    THINKING_PATTERN = r"<think>.*?<\/think>"
    ANSWER_PATTERN = r"<answer>.*?(\b[1-9]\d{0,3}(?:\s*,\s*[1-9]\d{0,3})*\b).*?<\/answer>"
    PARSING_REGEX = rf"{THINKING_PATTERN}\s*{ANSWER_PATTERN}"

    async def generate_alignments(
        self, client: ModelInterface, classes: list[dict[str, typ.Any]], **kwargs
    ) -> list[tuple[np.ndarray, np.ndarray, str]]:
        """Generate an alignment for a task."""
        request = self.format_request(client, classes=classes, **kwargs)
        response: list[BaseResponse] = await client.call(request=request)
        preds, probs, reasoning = self.compress_choices(response.choices, classes_size=len(classes) + 1)
        sparse_vector, probs_vector = self.parse_response(preds, probs, classes)
        return sparse_vector, probs_vector, reasoning

    def format_request(self, client: ModelInterface, classes: list[dict[str, typ.Any]], **kwargs) -> dict[str, typ.Any]:
        """Predict the true/false alignment."""
        prompt_template = self.format_prompt(classes=classes, **kwargs)
        prompt = self.prompt_messages_or_string(client, prompt_template)
        return {
            "prompt": prompt if client.endpoint == "completions" else None,
            "messages": prompt if client.endpoint == "chat/completions" else None,
            "seed": self.seed,
            "max_tokens": 5000,
            **self.sampling_params,
        }

    def compress_choices(self, choices: list[ResponseChoice], classes_size: int) -> tuple[list[int], list[float], str]:
        """Compress the choices into a prediction and probability by a majority vote and by averaging the probabilities.
        NOTE: Figure out how to average predictions across choices. Conformal prediction?
        """
        c = choices[0]
        answer_match = re.search(self.ANSWER_PATTERN, c.content)
        preds = [int(num.strip()) for num in answer_match.group(1).split(",")] if answer_match else []
        if not preds:
            logger.warning(f"Could not find any relevant tokens in the response: {c.content[-250:]}")
        probs = [0.0] * len(preds)
        return preds, probs, c.content


class StructuredLLMAligner(LLMAligner):
    """A LLM-based Alignment model applying long-context retrieval."""

    def __init__(self, max_attempts: int = 1, schema: pydantic.BaseModel = ReasoningModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_attempts = max_attempts
        self.schema = schema

    async def generate_alignments(
        self, client: ModelInterface, classes: list[dict[str, typ.Any]], **kwargs
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate an alignment for a task."""
        request = self.format_request(client, classes=classes, **kwargs)
        response: BaseResponse = await client.structured_call(
            request=request, schema=self.schema, max_attempts=self.max_attempts
        )

        if isinstance(response, (StructuredResponseError, type(None))):
            # If the response is an error, return a zero prediction
            logger.info("Could not parse the json response.")
            return np.zeros(len(classes))
        preds, probs = self.compress_choices(response.choices, classes_size=len(classes) + 1)
        sparse_vector, probs_vector = self.parse_response(preds, probs, classes)
        return sparse_vector, probs_vector

    def format_request(self, client: ModelInterface, classes: list[dict[str, typ.Any]], **kwargs) -> dict[str, typ.Any]:
        """Predict the true/false alignment."""
        prompt_template = self.format_prompt(classes=classes, **kwargs)
        prompt = self.prompt_messages_or_string(client, prompt_template)
        return {
            "prompt": prompt,
            "seed": self.seed,
            "stop": ["\n\n\n"],
            **self.sampling_params,
        }

    def compress_choices(self, choices: list[ResponseChoice], classes_size: int) -> tuple[list[int], list[float]]:
        """Compress the choices into a prediction and probability by a majority vote and by averaging the probabilities.
        NOTE: Figure out how to average predictions across choices. Conformal prediction?
        """
        c = choices[0]
        preds = c.validated_schema.code_ids or [  # type: ignore
            int(token) if self._is_predicted_alignment(token, classes_size) else 0.0
            for token in c.content.split()  # type: ignore
        ]
        probs = [0.0] * len(preds)
        if c.logprobs:
            logprobs = self._lookup_logprobs(c.logprobs, classes_size)
            preds = [int(token) if token.isdigit() else 0 for token in logprobs.tokens]
            if not preds:
                logger.info(
                    f"Could not find any relevant tokens in logprobs for the predicted tokens: {c.logprobs.tokens}."
                )
                logger.info("Using logprobs -0.01 instead.")
                preds = [0]
            top_logprobs = np.array([list(top_logprobs.values()) for top_logprobs in logprobs.top_logprobs])
            if top_logprobs.size == 0:
                logger.info(
                    f"Could not find any relevant tokens in top logprobs for the predicted tokens: {c.logprobs.tokens}."
                )
                logger.info("Using logprobs -0.01 instead.")
                top_logprobs = np.array([[-0.01] * len(preds)])
            norm_top_probs = self.normalized_exp_values(top_logprobs, axis=-1)
            probs = np.max(norm_top_probs, axis=-1)

        if len(preds) != len(probs):
            raise ValueError("Predictions and probabilities are not of the same length.")

        return preds, probs


class RegexLLMAligner(StructuredLLMAligner):
    """A LLM-based Alignment model applying long-context retrieval."""

    # LIST_REGEX_PATTERN = r"\[\s*\d+\s*(,\s*\d+\s*){0,20}\]"
    ICD_CODE_LIST = r"(?:[1-9]\d{0,3})(?:,(?:[1-9]\d{0,3})){0,19}\n"
    REASONING_PATTERN = r'Answer: [A-Z][\w \\",\\.]{30,1000}. The final selected ICD codes are: ' + ICD_CODE_LIST

    def __init__(self, regex_pattern: re.Pattern = ICD_CODE_LIST, max_attempts: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regex_pattern = regex_pattern
        self.max_attempts = max_attempts

    async def generate_alignments(
        self, client: ModelInterface, classes: list[dict[str, typ.Any]], **kwargs
    ) -> list[tuple[np.ndarray, np.ndarray, str]]:
        """Generate an alignment for a task."""
        request = self.format_request(client, classes=classes, **kwargs)
        response: list[BaseResponse] = await client.call(request=request)
        try:
            # icd_list = response.choices[0].content.split("The final ICD codes are: ")[-1]
            ast.literal_eval(response.choices[0].content)
        except ValueError:
            # If the response is an error, return a zero prediction
            logger.info("Could not parse the response list.")
            return np.zeros(len(classes)), np.zeros(len(classes))

        preds, probs = self.compress_choices(response.choices, classes_size=len(classes) + 1)
        sparse_vector, probs_vector = self.parse_response(preds, probs, classes)
        return sparse_vector, probs_vector, ""

    def format_request(self, client: ModelInterface, classes: list[dict[str, typ.Any]], **kwargs) -> dict[str, typ.Any]:
        """Predict the true/false alignment."""
        prompt_template = self.format_prompt(classes=classes, **kwargs)
        prompt = self.prompt_messages_or_string(client, prompt_template)
        return {
            "prompt": prompt,
            "guided_regex": self.regex_pattern,
            "seed": self.seed,
            "stop": ["\n"],
            **self.sampling_params,
        }

    def compress_choices(self, choices: list[ResponseChoice], classes_size: int) -> tuple[list[int], list[float]]:
        """Compress the choices into a prediction and probability by a majority vote and by averaging the probabilities.
        NOTE: Figure out how to average predictions across choices. Conformal prediction?
        """
        c = choices[0]
        # match comma separated list of integers with regex
        preds = [int(num.strip()) for num in c.content.split(",")]
        probs = [0.0] * len(preds)

        return preds, probs


def create_llm_aligner(
    aligner_type: str,
    prompt_name: str,
    sampling_params: dict[str, typ.Any],
    num_shots: int = 0,
    seed: int = 42,
    token_limit: int = 4048,
) -> LLMAligner:
    """
    Factory method to create an LLMAligner instance based on the specified type.

    Args:
        aligner_type (str): The type of aligner to create ("binary" or "long_context").
        prompt_name (str): The name of the prompt template to use.
        num_shots (int): Number of few-shot examples.
        token_limit (int): Token limit for the prompts.
        seed (int): Seed for random operations.
        sampling_params (dict[str, Any]): Sampling parameters for the LLM.

    Returns:
        LLMAligner: An instance of either BinaryLLMAligner or LongContextLLMAligner.
    """
    # if aligner_type == "binary":
    #     return BinaryLLMAligner(prompt_name, num_shots, token_limit, seed, sampling_params)
    if aligner_type == "structured":
        return StructuredLLMAligner(
            prompt_name=prompt_name,
            num_shots=num_shots,
            token_limit=token_limit,
            seed=seed,
            sampling_params=sampling_params,
        )
    elif aligner_type == "unconstrained":
        return UnconstrainedLLMAligner(
            prompt_name=prompt_name,
            num_shots=num_shots,
            token_limit=token_limit,
            seed=seed,
            sampling_params=sampling_params,
        )
    elif aligner_type == "regex":
        return RegexLLMAligner(
            prompt_name=prompt_name,
            num_shots=num_shots,
            token_limit=token_limit,
            seed=seed,
            sampling_params=sampling_params,
        )
    elif aligner_type == "cot-regex":
        return RegexLLMAligner(
            regex_pattern=RegexLLMAligner.REASONING_PATTERN,
            prompt_name=prompt_name,
            num_shots=num_shots,
            token_limit=token_limit,
            seed=seed,
            sampling_params=sampling_params,
        )
    else:
        raise ValueError(f"Unsupported aligner type: {aligner_type}")
