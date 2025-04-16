from functools import partial
import json
import re
import typing as typ
from loguru import logger

import numpy as np
import pydantic
from throughster.core.models import ResponseChoice, BaseResponse

from agents.base import HfBaseAgent
from dataloader.adapt.base import BaseModel
from dataloader.constants import PROJECT_ROOT
from trie import models

ANSWER_PATTERN = r"<answer>.*?(\b[1-9]\d{0,3}(?:\s*,\s*[1-9]\d{0,3})*\b).*?<\/answer>"


class InputModel(pydantic.BaseModel):
    """Input model for the Locate Agent."""

    note: str
    codes: list[models.Code]
    instructional_notes: list[models.Code]


class OutputModel(pydantic.BaseModel):
    """Output model for the Locate Agent."""

    codes: list[str]
    predictions: list[str]
    response: str | None = None

    @pydantic.field_validator("predictions")
    def check_predictions(cls, v: list[str]) -> list[str]:
        """Check the predictions."""
        if not v:
            return ["None"]
        return v


class MockAssignAgent(HfBaseAgent):
    """A dummy assign agent that simulates the candidate space"""

    def __init__(
        self,
        n_samples: int,
        per_code: bool = True,
        path_to_negatives: str = "data/medical-coding-systems/negatives/icd10cm_tabular_2022_negatives.json",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        with (PROJECT_ROOT / path_to_negatives).open() as f:
            negatives_data = json.load(f)
        self.negatives: dict[str, list] = negatives_data
        self.n_samples = n_samples
        self.per_code = per_code
        self.rng = np.random.RandomState(self.seed)

    async def _warmup(self, row: InputModel) -> None:
        """Warm up the model with a dummy request."""
        request = self.format_request(**row.model_dump())
        request["max_tokens"] = 1
        await self.client.call(request=request)

    async def predict(self, inputs: list[InputModel]) -> OutputModel:
        """Handle a batch of alignment tasks."""
        if len(inputs) > 1:
            await self._warmup(inputs[0])
        requests = [self.format_request(**el.model_dump()) for el in inputs]
        responses: list[BaseResponse] = await self.client.batch_call(requests=requests)
        response = ""
        predictions = set()

        for idx, resp in enumerate(responses):
            pred, reasoning = self.compress_choices(resp.choices)
            predictions.update([inputs[idx].codes[i - 1].name for i in pred])
            response += reasoning + "\n\n"

        return OutputModel(
            codes=[code.name for sublist in inputs for code in sublist.codes],
            predictions=list(predictions),
            response=response.strip(),
        )

    def _sample_negatives(self, code_id: str) -> list[str]:
        """Sample negatives for a given code."""
        population = self.negatives[code_id]
        weights = np.exp(-0.5 * np.arange(len(population)))  # Exponential decay
        weights /= weights.sum()
        return self.rng.choice(
            population, size=self.n_samples, replace=False, p=weights
        ).tolist()

    def _format_input(self, row: dict[str, typ.Any]) -> list[InputModel]:
        """Format the input."""
        m = BaseModel(**row)

        negative_ids: list[list[str]] = [
            self._sample_negatives(code) for code in m.targets
        ]
        positives: list[models.Code] = [
            self.trie.tabular[self.trie.lookup[code]] for code in m.targets
        ]
        negatives: list[list[models.Code]] = [
            [self.trie.tabular[self.trie.lookup[code]] for code in codes]
            for codes in negative_ids
        ]
        if self.per_code:
            codes = [
                sorted([pos] + negs, key=lambda c: c.name)
                for pos, negs in zip(positives, negatives)
            ]
            return [
                InputModel(note=m.note, codes=sublist, instructional_notes=sublist)
                for sublist in codes
            ]
        else:
            codes = sorted(
                [code for sublist in negatives for code in sublist] + positives,
                key=lambda c: c.name,
            )
            return [InputModel(note=m.note, codes=codes, instructional_notes=codes)]

    def compress_choices(self, choices: list[ResponseChoice]) -> tuple[list[int], str]:
        """Compress the choices."""
        c = choices[0]
        answer_match = re.search(ANSWER_PATTERN, c.content)
        preds = (
            [int(num.strip()) for num in answer_match.group(1).split(",")]
            if answer_match
            else []
        )
        if not preds:
            logger.warning(
                f"Could not find any relevant tokens in the response: {c.content[-250:]}"
            )
        return preds, c.content


def create_assign_agent(
    agent_type: str,
    prompt_name: str,
    sampling_params: dict[str, typ.Any],
    seed: int = 42,
) -> typ.Callable[..., HfBaseAgent]:
    """
    Factory method to create an AssignAgent instance based on the specified type.
    """
    if agent_type == "mock-per-code":
        return partial(
            MockAssignAgent,
            prompt_name=prompt_name,
            seed=seed,
            sampling_params=sampling_params,
            per_code=True,
        )
    elif agent_type == "mock":
        return partial(
            MockAssignAgent,
            prompt_name=prompt_name,
            seed=seed,
            sampling_params=sampling_params,
            per_code=False,
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
