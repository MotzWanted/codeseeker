from functools import partial
from itertools import chain
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


class Code(pydantic.BaseModel):
    """Model for a code."""

    name: str
    description: str | None = None
    etiology: bool
    manifestation: bool


class InputModel(pydantic.BaseModel):
    """Input model for the Locate Agent."""

    note: str
    codes: list[Code]
    instructional_notes: list[models.InstructionalNote]


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
        path_to_negatives: str = "data/medical-coding-systems/negatives/icd10cm_2022_negatives.json",
        per_code: bool = False,
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
        self.weights = np.exp(-0.5 * np.arange(100))
        self.weights /= self.weights.sum()

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
        """Sample negatives for a given code (with capped population)."""
        population = self.negatives[code_id][:100]  # Cap to first 100 items
        n = min(len(population), self.n_samples)
        if n == 0:
            return []
        return self.rng.choice(
            population, size=n, replace=False, p=self.weights
        ).tolist()

    def _code_from_trie(self, code: str) -> Code:
        trie_entry = self.trie[code]
        return Code(
            name=trie_entry.name,
            description=trie_entry.description,
            etiology=trie_entry.etiology,  # type: ignore
            manifestation=trie_entry.manifestation,  # type: ignore
        )

    def _to_code(self, codes: list[str]) -> list[Code]:
        return [self._code_from_trie(c) for c in codes]

    def _format_input(self, row: dict[str, typ.Any]) -> list[InputModel]:
        """Format the input."""
        m = BaseModel(**row)

        negative_ids = [self._sample_negatives(code) for code in m.targets]
        positives = self._to_code(m.targets)
        negatives: list[list[Code]] = [self._to_code(ids) for ids in negative_ids]
        if self.per_code:
            # Per-code grouping of positive + negatives
            inputs = []
            for pos, neg_group in zip(positives, negatives):
                group = [pos] + neg_group
                group.sort(key=lambda c: c.name)
                code_names = [c.name for c in group]
                instructions = self.trie.get_instructional_notes(codes=code_names)
                inputs.append(
                    InputModel(
                        note=m.note, codes=group, instructional_notes=instructions
                    )
                )
            return inputs

        else:
            all_codes = sorted(
                list(chain.from_iterable(negatives)) + positives, key=lambda c: c.name
            )
            code_names = [c.name for c in all_codes]
            instruction_notes = self.trie.get_instructional_notes(codes=code_names)
            return [
                InputModel(
                    note=m.note, codes=all_codes, instructional_notes=instruction_notes
                )
            ]

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

    @staticmethod
    def sample_negatives(
        negatives: list[list[str]],
        positives: list[str],
        per_positive: int,
        seed: int | str = 42,
    ) -> list[str]:
        # Ensure the total number of samples does not exceed 50
        negatives_to_sample = len(positives) * per_positive
        if negatives_to_sample == 0:
            return []
        rng = np.random.RandomState(int(seed))
        # Step 1: Determine the initial fair share per sublist
        num_sublists = len(negatives)
        if num_sublists == 0:
            raise ValueError("No negatives to sample from")
        base_samples_per_sublist = negatives_to_sample // num_sublists
        remainder = negatives_to_sample % num_sublists  # Leftover samples

        selected_negatives = []
        remaining_negatives = negatives_to_sample

        # Step 2: Assign samples as evenly as possible
        for i, sublist in enumerate(sorted(negatives)):
            if remaining_negatives <= 0:
                break
            positive_set = set(positives)
            selected_set = set(selected_negatives)

            unique_sublist = [
                code
                for code in sublist
                if code not in positive_set and code not in selected_set
            ]

            if len(unique_sublist) == 0:
                continue

            indices = np.arange(len(unique_sublist))
            weights = np.exp(-0.5 * indices)  # Exponential decay
            weights /= weights.sum()  # Normalize to get probabilities

            num_to_sample = min(
                len(unique_sublist),
                base_samples_per_sublist + (1 if i < remainder else 0),
            )
            sampled = rng.choice(
                unique_sublist, size=num_to_sample, replace=False, p=weights
            ).tolist()

            selected_negatives.extend(sampled)
            remaining_negatives -= len(sampled)

        if len(selected_negatives) != negatives_to_sample:
            raise ValueError(
                f"Sampled {len(selected_negatives)} negatives, but expected {negatives_to_sample}"
            )

        return selected_negatives


def create_assign_agent(
    agent_type: str,
    prompt_name: str,
    sampling_params: dict[str, typ.Any],
    seed: int = 42,
) -> typ.Callable[..., HfBaseAgent]:
    """
    Factory method to create an AssignAgent instance based on the specified type.
    """
    if agent_type == "mock":
        return partial(
            MockAssignAgent,
            prompt_name=prompt_name,
            seed=seed,
            sampling_params=sampling_params,
            per_code=False,
        )
    elif agent_type == "mock-single":
        return partial(
            MockAssignAgent,
            prompt_name=prompt_name,
            seed=seed,
            sampling_params=sampling_params,
            per_code=True,
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
