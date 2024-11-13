import abc
import json
from pathlib import Path
import random
import typing as typ

from loguru import logger
import numpy as np
from prompt_poet import Prompt

from alignment.base import Aligner, matrix2list
from alignment.models import Alignment, AlignmentSingleton

from throughster import ModelInterface
from throughster.core.models import BaseResponse, LogProbs, ResponseChoice
from throughster.vllm.client import VllmOpenAiInterface

PATH_TO_TEMPLATES = Path(__file__).parent / "templates"


def custom_tojson(value):
    # Use json.dumps with ensure_ascii=False to avoid unnecessary escaping
    return json.dumps(value, ensure_ascii=False)


class LLMAligner(Aligner):
    """A LLM-based alignment model"""

    def __init__(
        self, prompt_name: str, num_shots: int, token_limit: int, seed: int, sampling_params: dict[str, typ.Any]
    ):
        self.template_path = PATH_TO_TEMPLATES / f"{prompt_name}.yml.j2"
        self.prompt_name = prompt_name
        self.num_shots = num_shots
        self.token_limit = token_limit
        self.seed = seed
        self.sampling_params = sampling_params

    async def predict(
        self, client: ModelInterface, corpus: list[str], queries: list[str], fewshots: list | None = [], **kwargs
    ) -> Alignment:
        """Align a list of queries elements to a corpus."""
        fewshots = fewshots or []

        results = await self.generate_alignments(client=client, corpus=corpus, queries=queries, fewshots=fewshots)

        sparse_matrix = np.zeros((len(queries), len(corpus)))
        probs_matrix = np.zeros((len(queries), len(corpus)))
        for i, r in enumerate(results):
            sparse_vector, probs_vector = r
            sparse_matrix[i] = sparse_vector
            probs_matrix[i] = probs_vector

        indexes = matrix2list(sparse_matrix)

        return Alignment(indexes=indexes, matrix=sparse_matrix, probabilities=probs_matrix)

    @abc.abstractmethod
    async def generate_alignments(
        self, client: ModelInterface, corpus: list[str], queries: list[str], fewshots: list[dict[str, typ.Any]]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        ...

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
            template_path=self.template_path,
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


# class BinaryLLMAligner(LLMAligner):
#     """A LLM-based Alignment model applying binary classifications."""

#     async def generate_alignments(
#         self, client: ModelInterface, corpus: list[str], queries: list[str], fewshots: list[dict[str, typ.Any]]
#     ) -> typ.AsyncGenerator[tuple[np.ndarray, np.ndarray], None]:
#         """Generate an alignment for a task."""
#         async with client as c:
#             results: list[BaseResponse] = await asyncio.gather(
#                 *[
#                     self.predict_true_false(client=c, passage=passage, fact=query, fewshots=fewshots)  # noqa: F821
#                     for passage in corpus
#                 ]
#             )
#         sparse_vector = np.zeros(len(corpus))
#         probs_vector = np.zeros(len(corpus))
#         for i, r in enumerate(results):
#             pred, prob = self.compress_choices(r.choices)
#             sparse_vector[i] = pred
#             probs_vector[i] = prob
#         return sparse_vector, probs_vector

#     async def predict_true_false(
#         self, client: ModelInterface, passage: str, fact: str, fewshots: list[dict[str, typ.Any]]
#     ) -> BaseResponse:
#         """Predict the true/false alignment."""
#         prompt_template = self.format_prompt_with_shots(fewshots, passage=passage, fact=fact)
#         prompt = self.prompt_messages_or_string(client, prompt_template)
#         guided_choices = ["True", "False"]
#         request = {
#             "prompt": prompt,
#             "logprobs": 2,
#             "guided_choice": guided_choices,
#             "max_tokens": 1,
#             "seed": self.seed,
#             **self.sampling_params,
#         }
#         return await client.call(request)

#     def compress_choices(self, choices: list[ResponseChoice]) -> tuple[float, float]:
#         """Compress the choices into a prediction and probability by a majority vote and by averaging."""
#         predictions = np.zeros(len(choices))
#         probs = np.zeros(len(choices))
#         for i, c in enumerate(choices):
#             predictions[i] = 1 if c.content == "True" else 0
#             if c.logprobs is None:
#                 continue
#             top_logprobs = np.array(list(c.logprobs.top_logprobs[0].values()))
#             norm_top_logprobs = self.normalized_exp_values(top_logprobs, axis=-1)
#             probs[i] = np.max(norm_top_logprobs)
#         majority_pred = np.round(np.mean(predictions), 0)
#         avg_prob = np.round(np.mean(probs), 4)
#         return majority_pred, avg_prob


class StructuredLLMAligner(LLMAligner):
    """A LLM-based Alignment model applying long-context retrieval."""

    def __init__(self, max_attempts: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_attempts = max_attempts

    async def predict(
        self, client: ModelInterface, corpus: list[str], queries: list[str], fewshots: list | None = [], **kwargs
    ) -> Alignment:
        """Align a list of corpus elements to a set of queries."""
        fewshots = fewshots or []
        if isinstance(client, VllmOpenAiInterface):
            # Activate prefix caching in KV cache.
            request = self.format_request(client=client, query=queries[0], corpus=corpus, fewshots=fewshots)
            await client.structured_call(request=request, schema=AlignmentSingleton, max_attempts=1)
        return await super().predict(client=client, corpus=corpus, queries=queries, fewshots=fewshots)

    async def generate_alignments(
        self, client: ModelInterface, corpus: list[str], queries: list[str], fewshots: list[dict[str, typ.Any]]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate an alignment for a task."""
        requests = [self.format_request(client, corpus=corpus, query=q, fewshots=fewshots) for q in queries]
        responses: list[BaseResponse] = await client.structured_batch_call(
            requests=requests, schema=AlignmentSingleton, max_attempts=self.max_attempts
        )

        results = []
        for response in responses:
            sparse_vector = np.zeros(len(corpus))
            probs_vector = np.zeros(len(corpus))
            preds, probs = self.compress_choices(response.choices, corpus_size=len(corpus) + 1)
            for i, _ in enumerate(corpus, start=1):
                if i not in preds:
                    continue
                sparse_vector[i - 1] = 1
                probs_vector[i - 1] = probs[preds.index(i)]
            results.append((sparse_vector, probs_vector))
        return results

    def format_request(
        self, client: ModelInterface, query: str, corpus: list[str], fewshots: list[dict[str, typ.Any]]
    ) -> dict[str, typ.Any]:
        """Predict the true/false alignment."""
        prompt_template = self.format_prompt_with_shots(fewshots, query=query, corpus=corpus)
        prompt = self.prompt_messages_or_string(client, prompt_template)
        return {
            "prompt": prompt,
            "seed": self.seed,
            "stop": ["\n\n\n"],
            **self.sampling_params,
        }

    def compress_choices(self, choices: list[ResponseChoice], corpus_size: int) -> tuple[list[int], list[float]]:
        """Compress the choices into a prediction and probability by a majority vote and by averaging the probabilities.
        NOTE: Figure out how to average predictions across choices. Conformal prediction?
        """
        c = choices[0]
        preds = c.validated_schema.ids or [  # type: ignore
            int(token) if self._is_predicted_alignment(token, corpus_size) else 0.0
            for token in c.content.split()  # type: ignore
        ]
        probs = [0.0] * len(preds)
        if c.logprobs:
            logprobs = self._lookup_logprobs(c.logprobs, corpus_size)
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
    def _is_predicted_alignment(token: str, corpus_size: int) -> bool:
        return (token.isdigit() and int(token) < corpus_size) or "[]" in token

    def _lookup_logprobs(self, logprobs: LogProbs, corpus_size: int) -> LogProbs:
        """Based on a list of tokens we need to find logprobs of those tokens that represent the alignment predictions.
        We are looking for numeric tokens and the token "[]" which represents the absence of an alignment ("[0]").
        """
        _logprobs = logprobs.model_copy()
        valid_indices = [
            i for i, token in enumerate(logprobs.tokens) if self._is_predicted_alignment(token, corpus_size)
        ]

        _logprobs.text_offset = [logprobs.text_offset[i] for i in valid_indices]
        _logprobs.token_logprobs = [logprobs.token_logprobs[i] for i in valid_indices]
        _logprobs.tokens = [logprobs.tokens[i] for i in valid_indices]
        _logprobs.top_logprobs = [
            self._lookup_top_logprobs(logprobs.top_logprobs[i], corpus_size) for i in valid_indices
        ]

        return _logprobs

    def _lookup_top_logprobs(self, top_logprobs: dict[str, float], corpus_size: int) -> dict[str, float]:
        return {k: v for k, v in top_logprobs.items() if self._is_predicted_alignment(k, corpus_size)}


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
    else:
        raise ValueError(f"Unsupported aligner type: {aligner_type}")
