import abc
import ast
import json
from pathlib import Path
import random
import re
import typing as typ

from jinja2 import Environment, FileSystemLoader
from loguru import logger
import numpy as np
from prompt_poet import Prompt

from alignment.base import Aligner, matrix2list
from alignment.models import Alignment, AlignmentSingleton

from throughster import ModelInterface
from throughster.core.models import BaseResponse, LogProbs, ResponseChoice
from throughster.vllm.client import VllmOpenAiInterface
from throughster.core.errors import StructuredResponseError

PATH_TO_TEMPLATES = Path(__file__).parent / "templates"


def custom_tojson(value):
    # Use json.dumps with ensure_ascii=False to avoid unnecessary escaping
    return json.dumps(value, ensure_ascii=False)


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

    async def predict(
        self, client: ModelInterface, classes: list[str], segments: list[str], fewshots: list | None = [], **kwargs
    ) -> Alignment:
        """Align a list of segments elements to a classes."""
        fewshots = fewshots or []

        results = await self.generate_alignments(client=client, classes=classes, segments=segments, fewshots=fewshots)

        sparse_matrix = np.zeros((len(segments), len(classes)))
        probs_matrix = np.zeros((len(segments), len(classes)))
        for i, r in enumerate(results):
            sparse_vector, probs_vector = r
            sparse_matrix[i] = sparse_vector
            probs_matrix[i] = probs_vector

        indexes = matrix2list(sparse_matrix)

        return Alignment(indexes=indexes, matrix=sparse_matrix, probabilities=probs_matrix)

    @abc.abstractmethod
    async def generate_alignments(
        self, client: ModelInterface, classes: list[str], segments: list[str], fewshots: list[dict[str, typ.Any]]
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

    def format_prompt(self, template_data: dict[str, typ.Any], truncation_step: int = 1) -> Prompt:
        """Format the prompt."""
        return Prompt(
            raw_template=self.raw_template,
            template_data=template_data,
            token_limit=self.token_limit,
            truncation_step=truncation_step,
        )

    def format_prompt_with_shots(self, fewshots: list[dict[str, typ.Any]], **kwargs) -> Prompt:
        """Format the prompt."""
        random.seed(self.seed)
        template_data = {"fewshots": random.sample(fewshots, self.num_shots), "custom_tojson": custom_tojson, **kwargs}
        prompt = self.format_prompt(template_data)
        if fewshots:  # maybe truncate fewshots examples
            prompt.tokenize()
            if prompt._token_limit < prompt._total_tokens:
                prompt.truncate()
        return prompt

    @staticmethod
    def prompt_messages_or_string(client: ModelInterface, prompt: Prompt) -> str | list[dict[str, str]]:
        if client.endpoint == "chat/completions":
            return prompt.messages
        return prompt.string


class StructuredLLMAligner(LLMAligner):
    """A LLM-based Alignment model applying long-context retrieval."""

    def __init__(self, max_attempts: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_attempts = max_attempts

    async def predict(
        self, client: ModelInterface, classes: list[str], segments: list[str], fewshots: list | None = [], **kwargs
    ) -> Alignment:
        """Align a list of classes elements to a set of segments."""
        fewshots = fewshots or []
        if isinstance(client, VllmOpenAiInterface):
            # Activate prefix caching in KV cache.
            request = self.format_request(client=client, segment=segments[0], classes=classes, fewshots=fewshots)
            await client.call(request=request)
        return await super().predict(client=client, classes=classes, segments=segments, fewshots=fewshots)

    async def generate_alignments(
        self, client: ModelInterface, classes: list[str], segments: list[str], fewshots: list[dict[str, typ.Any]]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate an alignment for a task."""
        requests = [self.format_request(client, classes=classes, segment=s, fewshots=fewshots) for s in segments]
        responses: list[BaseResponse] = await client.structured_batch_call(
            requests=requests, schema=AlignmentSingleton, max_attempts=self.max_attempts
        )
        results = []
        for response in responses:
            if isinstance(response, (StructuredResponseError, type(None))):
                # If the response is an error, return a zero prediction
                results.append((np.zeros(len(classes)), np.zeros(len(classes))))
                continue
            preds, probs = self.compress_choices(response.choices, classes_size=len(classes) + 1)
            sparse_vector, probs_vector = self.parse_response(preds, probs, classes)
            results.append((sparse_vector, probs_vector))
        return results

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

    def format_request(
        self, client: ModelInterface, segment: str, classes: list[str], fewshots: list[dict[str, typ.Any]]
    ) -> dict[str, typ.Any]:
        """Predict the true/false alignment."""
        prompt_template = self.format_prompt_with_shots(fewshots, segment=segment, classes=classes)
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
        preds = c.validated_schema.ids or [  # type: ignore
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

    @staticmethod
    def _is_predicted_alignment(token: str, classes_size: int) -> bool:
        return (token.isdigit() and int(token) < classes_size) or "[]" in token

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


class RegexLLMAligner(StructuredLLMAligner):
    """A LLM-based Alignment model applying long-context retrieval."""

    # LIST_REGEX_PATTERN = r"\[\s*\d+\s*(,\s*\d+\s*){0,20}\]"
    ICD_CODE_LIST = r"\[\d+(,\d+){0,19}\]\n"
    REASONING_PATTERN = r'Answer: T[\w \\",\\.]{30,250}. The final ICD codes are: ' + ICD_CODE_LIST

    def __init__(self, regex_pattern: re.Pattern = ICD_CODE_LIST, max_attempts: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regex_pattern = regex_pattern
        self.max_attempts = max_attempts

    async def generate_alignments(
        self, client: ModelInterface, classes: list[str], segments: list[str], fewshots: list[dict[str, typ.Any]]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate an alignment for a task."""
        requests = [self.format_request(client, classes=classes, segment=s, fewshots=fewshots) for s in segments]
        responses: list[BaseResponse] = await client.batch_call(requests=requests)
        results = []
        for response in responses:
            try:
                icd_list = response.choices[0].content.split("The final ICD codes are: ")[-1]
                ast.literal_eval(icd_list)
            except ValueError:
                # If the response is an error, return a zero prediction
                logger.info("Could not parse the response list.")
                results.append((np.zeros(len(classes)), np.zeros(len(classes))))
                continue
            preds, probs = self.compress_choices(response.choices, classes_size=len(classes) + 1)
            sparse_vector, probs_vector = self.parse_response(preds, probs, classes)
            results.append((sparse_vector, probs_vector))
        return results

    def format_request(
        self, client: ModelInterface, segment: str, classes: list[str], fewshots: list[dict[str, typ.Any]]
    ) -> dict[str, typ.Any]:
        """Predict the true/false alignment."""
        prompt_template = self.format_prompt_with_shots(fewshots, segment=segment, classes=classes)
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
        icd_list = c.content.split("The final ICD codes are: ")[-1]
        preds = ast.literal_eval(icd_list)
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


class StructuredLLMSelfAligner(StructuredLLMAligner):
    """A LLM-based Alignment model applying long-context retrieval."""

    def __init__(self, max_attempts: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_attempts = max_attempts

    async def generate_alignments(
        self, client: ModelInterface, classes: list[str], segments: list[str], fewshots: list[dict[str, typ.Any]]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate an alignment for a task."""
        results = []
        extended_fewshots = fewshots.copy()
        for s in segments:
            request = self.format_request(client, classes=classes, segment=s, fewshots=extended_fewshots)
            try:
                response: BaseResponse = await client.structured_call(request=request, schema=AlignmentSingleton)
                preds, probs = self.compress_choices(response.choices, classes_size=len(classes) + 1)
                sparse_vector, probs_vector = self.parse_response(preds, probs, classes)
                results.append((sparse_vector, probs_vector))
            except StructuredResponseError:
                results.append((np.zeros(len(classes)), np.zeros(len(classes))))
                continue
            # Add the predictions to the fewshots for the next segment
            extended_fewshots.append({"segment": s, "labels": preds})
        return results

    def format_request(
        self, client: ModelInterface, segment: str, classes: list[str], fewshots: list[dict[str, typ.Any]]
    ) -> dict[str, typ.Any]:
        """Predict the true/false alignment."""
        prompt_template = self.format_prompt_with_shots(fewshots, segment=segment, classes=classes)
        prompt = self.prompt_messages_or_string(client, prompt_template)
        return {
            "prompt": prompt,
            "seed": self.seed,
            "stop": ["\n\n\n"],
            **self.sampling_params,
        }


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
    elif aligner_type == "regex":
        return RegexLLMAligner(
            prompt_name=prompt_name,
            num_shots=num_shots,
            token_limit=token_limit,
            seed=seed,
            sampling_params=sampling_params,
        )
    elif aligner_type == "structured_self":
        return StructuredLLMSelfAligner(
            prompt_name=prompt_name,
            num_shots=num_shots,
            token_limit=token_limit,
            seed=seed,
            sampling_params=sampling_params,
        )
    else:
        raise ValueError(f"Unsupported aligner type: {aligner_type}")
