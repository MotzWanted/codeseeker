import abc
from collections import defaultdict
import datasets
import numpy as np
import rich
import typing as typ

from dataloader.adapters.alignment import (
    AlignmentModel,
    AlignmentModelForTraining,
    NbmeAdapter,
    SyntheticAlignmentModel,
)
from dataloader.loaders.nbme.nbme_notes import NbmeDatasetLoader
from segmenters.base import Segmenter, factory


class AlignmentAdapterForTraining(abc.ABC):
    """Adapter for alignment instances associated with a single query.
    NOTE: used for formatting the data for training the alignment model."""

    @abc.abstractmethod
    def is_compatible(self, row: dict[str, typ.Any]) -> bool:
        """Check if a row is compatible with this adapter."""
        ...

    @abc.abstractmethod
    def adapt(self, row: dict[str, typ.Any]) -> list[AlignmentModelForTraining]:
        """Adapt a `SyntheticAlignmentModel` to `AlignmentModelForTraining` by flattening queries."""
        ...

    def __call__(self, batch: dict[str, list[typ.Any]], **extras: typ.Any) -> dict[str, list[typ.Any]]:
        """Adapt the row and add extra fields."""
        batch_size = len(batch[list(batch.keys())[0]])
        output = defaultdict(list)
        for idx in range(batch_size):
            row = {key: batch[key][idx] for key in batch}
            output_rows: typ.Sequence[AlignmentModelForTraining] = self.adapt(row)
            for row in output_rows:
                for key, value in row.model_dump().items():
                    output[key].append(value)
                output["extras"].append(extras)
        return output


class Test2Training(AlignmentAdapterForTraining):
    """Adapter for alignment instances associated with a single query.
    NOTE: used for formatting the data for training the alignment model."""

    input_model: typ.Type[AlignmentModel] = AlignmentModel

    def is_compatible(self, row: dict[str, typ.Any]) -> bool:
        """Check if a row is compatible with this adapter."""
        return self.input_model.model_validate(row) is not None

    def adapt(self, row: dict[str, typ.Any]) -> list[AlignmentModelForTraining]:
        """Adapt a `SyntheticAlignmentModel` to `AlignmentModelForTraining` by flattening queries."""
        struct_row: AlignmentModel = self.input_model(**row)
        return [
            AlignmentModelForTraining(
                aid=f"{struct_row.aid}-{idx}",
                entities=struct_row.entities,
                segment=segment,
                targets=label,
                probabilities=None,
            )
            for idx, (segment, label) in enumerate(zip(struct_row.segments, struct_row.labels))
        ]


class Synthetic2Training(AlignmentAdapterForTraining):
    """Adapter for alignment instances associated with a single query.
    NOTE: used for formatting the data for training the alignment model."""

    input_model: typ.Type[SyntheticAlignmentModel] = SyntheticAlignmentModel

    def is_compatible(self, row: dict[str, typ.Any]) -> bool:
        """Check if a row is compatible with this adapter."""
        return self.input_model.model_validate(row) is not None

    def adapt(self, row: dict[str, typ.Any]) -> list[AlignmentModelForTraining]:
        """Adapt a `SyntheticAlignmentModel` to `AlignmentModelForTraining` by flattening queries."""
        struct_row: SyntheticAlignmentModel = self.input_model(**row)
        return [
            AlignmentModelForTraining(
                aid=f"{struct_row.aid}-{idx}",
                entities=struct_row.entities,
                segment=segment,
                targets=prediction,
                probabilities=[p for p in probs if p > 0.0],
            )
            for idx, (segment, prediction, probs) in enumerate(
                zip(struct_row.segments, struct_row.predictions, struct_row.probabilities)
            )
        ]


def _get_dataset(dset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    """Get a `datasets.Dataset`."""
    if isinstance(dset, datasets.Dataset):
        return dset
    return next(iter(dset.values()))


NUM_WORKERS = 16
dump_folder = "/Users/amo/research/patient-note-alignment/synthetic-data/Meta-Llama-3.1-70B-instruct/v2"
dataset_path = "/Users/amo/research/patient-note-alignment/synthetic-data/Meta-Llama-3.1-70B-instruct/nbme-train-subset/be11b9a08c322ded264e4f365675f3b1/synthetic_data/"
full_train: datasets.Dataset = datasets.load_from_disk(dataset_path)  # type: ignore
test: datasets.Dataset = NbmeDatasetLoader().load_dataset(split="test", num_proc=NUM_WORKERS)  # type: ignore
segmenter: Segmenter = factory("nbme", "en_core_web_lg")  # noqa: F821
adapter = NbmeAdapter(segmenter=segmenter)
test = test.map(
    adapter,
    num_proc=NUM_WORKERS,
    desc=f"Adapting dataset to `AlignmentModel` using `{NbmeAdapter.__name__}`.",
    remove_columns=_get_dataset(test).column_names,
)

validation_indices = []
train_indices = []
for idx, probs in enumerate(full_train["probabilities"]):
    if np.all(np.array(probs)[np.array(probs) > 0.0] == 1.0):
        validation_indices.append(idx)
    else:
        train_indices.append(idx)

validation = full_train.select(validation_indices)
train = full_train.select(train_indices)

adapted_train = train.map(
    Synthetic2Training(),
    num_proc=NUM_WORKERS,
    batched=True,
    batch_size=1,
    desc="Adapting train to `AlignmentModelForTraining`...",
    remove_columns=_get_dataset(train).column_names,
)
adapted_validation = validation.map(
    Synthetic2Training(),
    num_proc=NUM_WORKERS,
    batched=True,
    batch_size=1,
    desc="Adapting validation to `AlignmentModelForTraining`...",
    remove_columns=_get_dataset(validation).column_names,
)
adapted_test = test.map(
    Test2Training(),
    num_proc=NUM_WORKERS,
    batched=True,
    batch_size=1,
    desc="Adapting test to `AlignmentModelForTraining`...",
    remove_columns=_get_dataset(test).column_names,
)

merged_data = datasets.DatasetDict({"train": adapted_train, "validation": adapted_validation, "test": adapted_test})
rich.print(merged_data)
merged_data.save_to_disk(dump_folder)
