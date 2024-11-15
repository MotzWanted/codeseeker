import asyncio
from collections import defaultdict
import time
import typing as typ


from alignment.base import Aligner, list2matrix
from throughster.base import ModelInterface
from throughster.hf_datasets import HfOperation
from alignment.models import Alignment
from tools.exception import dump_exceptions_to_file


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

        responses = asyncio.run(self._async_call_wrapper(batch_rows))

        output = defaultdict(list)
        for i, r in enumerate(responses):
            alignment_data = r.model_dump()
            for key, value in alignment_data.items():
                output[key].append(value)

            labels_matrix = None
            if "labels" in batch and batch["labels"][i]:
                labels_matrix = list2matrix(len(batch["segments"][i]), len(batch["entities"][i]), batch["labels"][i])
            output["labels_matrix"].append(labels_matrix)

        return {
            **batch,
            **output,
        }

    async def _async_call_wrapper(self, batch_rows: list[dict[str, typ.Any]]) -> list[Alignment]:
        """Handle a batch of alignment tasks."""
        return await asyncio.gather(*[self.aligner.predict(client=self.client, **row) for row in batch_rows])

    def _validate_input(self, batch: dict[str, list[typ.Any]]) -> None:
        """Validate the input batch."""
        for key in ["entities", "segments"]:
            if key not in batch:
                raise ValueError(f"Missing key: {key}. Available keys: {batch.keys()}")
            if key and not isinstance(batch[key], list):
                raise ValueError(f"Invalid type for key {key}: {type(batch[key])}")


class HfSyntheticAlignment(HfOperation):
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

    @dump_exceptions_to_file
    def __call__(self, row: dict[str, typ.Any], *args, **kwargs) -> dict[str, list[typ.Any]]:
        """Process a row of alignment tasks from a HuggingFace datasets.map()."""
        self._validate_input(row)

        if "labels" in row and row["labels"]:
            sparse_matrix = list2matrix(len(row["segments"]), len(row["entities"]), row["labels"])
            alignment_data = Alignment(indexes=row["labels"], matrix=sparse_matrix, probabilities=sparse_matrix)
            return {
                **row,
                "predictions": alignment_data.indexes,
                "probabilities": alignment_data.probabilities,
                "sparse_matrix": alignment_data.matrix,
            }

        responses = asyncio.run(self._async_call_wrapper(row))
        time.sleep(self.wait_time)

        alignment_data = typ.cast(Alignment, responses[0])

        return {
            **row,
            "predictions": alignment_data.indexes,
            "probabilities": alignment_data.probabilities,
            "sparse_matrix": alignment_data.matrix,
        }

    async def _async_call_wrapper(self, row: dict[str, typ.Any]) -> list[Alignment]:
        """Handle a batch of alignment tasks."""
        return await asyncio.gather(*[self.aligner.predict(client=self.client, **row)])

    def _validate_input(self, batch: dict[str, list[typ.Any]]) -> None:
        """Validate the input batch."""
        for key in ["entities", "segments"]:
            if key not in batch:
                raise ValueError(f"Missing key: {key}. Available keys: {batch.keys()}")
            if key and not isinstance(batch[key], list):
                raise ValueError(f"Invalid type for key {key}: {type(batch[key])}")
