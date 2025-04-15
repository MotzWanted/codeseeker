import asyncio
from collections import defaultdict
import typing as typ

import numpy as np


from agents.base import Aligner
from throughster.base import ModelInterface
from throughster.hf_datasets import HfOperation
from agents.models import Alignment
from dataloader.adapt.base import BaseModel


class HfAlignment(HfOperation):
    def __init__(
        self,
        *,
        init_client_fn: typ.Callable[..., ModelInterface],
        aligner: Aligner,
        wait_time: int = 0,
    ):
        self.init_client_fn = init_client_fn
        self.aligner = aligner
        self.wait_time = wait_time

    def __call__(self, batch: dict[str, list[typ.Any]], *args, **kwargs) -> dict[str, list[typ.Any]]:
        """Process a row of alignment tasks from a HuggingFace datasets.map()."""
        self._validate_input(batch)
        batch_size = len(batch[list(batch.keys())[0]])

        batch_rows = [{key: value[i] for key, value in batch.items()} for i in range(batch_size)]
        batch_rows = [self._format_targets(row) for row in batch_rows]

        responses = asyncio.run(self._async_call_wrapper(batch_rows))

        # Override batch with batch_rows reformatted as a dict of lists
        batch = {key: [row[key] for row in batch_rows] for key in batch_rows[0].keys()}

        output = defaultdict(list)
        for i, r in enumerate(responses):
            alignment_data = r.model_dump()
            for key, value in alignment_data.items():
                if isinstance(value, np.ndarray):
                    if np.any(value is None):  # or use np.isnan if float
                        raise ValueError(f"None value found in array at key {key}")
                    output[key].append(value.tolist())
                else:
                    output[key].append(value)

        return {
            **batch,
            **output,
        }

    @staticmethod
    def _format_targets(rows: dict[str, typ.Any]) -> dict[str, typ.Any]:
        """Format the targets."""
        m = BaseModel(**rows)
        parsed_targets = m.parse_targets()
        return {**m.model_dump(), "targets": parsed_targets, "codes": m.targets}

    async def _async_call_wrapper(self, batch_rows: list[dict[str, typ.Any]]) -> list[Alignment]:
        """Handle a batch of alignment tasks."""
        return await asyncio.gather(*[self.aligner.predict(client=self.client, **row) for row in batch_rows])

    def _validate_input(self, batch: dict[str, list[typ.Any]]) -> None:
        """Validate the input batch."""
        for key in ["classes", "targets"]:
            if key not in batch:
                raise ValueError(f"Missing key: {key}. Available keys: {batch.keys()}")
            if key and not isinstance(batch[key], list):
                raise ValueError(f"Invalid type for key {key}: {type(batch[key])}")
