from __future__ import annotations

import typing

import datasets
from pydantic_settings import BaseSettings, SettingsConfigDict
import rich

from dataloader import meddec, snomed, mdace_inpatient
import dataloader
from dataloader.base import DatasetConfig
from dataloader import mimiciv
from segmenters.base import factory


def my_loader(
    subset: str | None = None,
    split: str | None = None,
    **kws: dict[str, typing.Any],
) -> datasets.Dataset:
    """Define a custom data loader."""
    return datasets.Dataset.from_list(
        [
            {
                "question": "What is the meaning of life?",
                "answer": "42",
            },
        ]
    )


SEGMENTER = factory("document", spacy_model="en_core_web_lg")

DATASET_CONFIGS = {
    "meddec": {
        "identifier": "meddec",
        "name_or_path": meddec,
        "split": "train",
        "options": {"segmenter": SEGMENTER},
    },
    "snomed": {
        "identifier": "snomed",
        "name_or_path": snomed,
        "split": "train",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-diagnosis-3": {
        "identifier": "mdace-diagnosis-3",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-3"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-procedure-4": {
        "identifier": "mdace-procedure-4",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10pcs-4"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-icd10cm-3.0": {
        "identifier": "mdace-icd10cm-3.0",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-3.0"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-icd10cm-3.1": {
        "identifier": "mdace-icd10cm-3.1",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-3.1"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-icd10cm-3.2": {
        "identifier": "mdace-icd10cm-3.2",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-3.2"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-icd10cm-3.3": {
        "identifier": "mdace-icd10cm-3.3",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-3.3"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-icd10cm-3.4": {
        "identifier": "mdace-icd10cm-3.4",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-3.4"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mimic-iv": {
        "identifier": "mimic-iv",
        "name_or_path": mimiciv,
        "split": "test",
        "subsets": ["icd10-3.4"],
        "options": {"segmenter": SEGMENTER, "negatives": 100, "hard_negatives": 1.0},
    },
    "my_data": {
        "identifier": "my_data",
        "name_or_path": my_loader,
        "path": my_loader,
    },
}


class Arguments(BaseSettings):
    """Arguments for the script."""

    name: str = "mimic-iv"

    model_config = SettingsConfigDict(cli_parse_args=True, frozen=True)


def run(args: Arguments) -> None:
    """Showcase the `load_dataset` function."""
    try:
        config = DatasetConfig(**DATASET_CONFIGS[args.name])
    except KeyError as exc:
        raise KeyError(f"Configuration for `{args.name}` not found!") from exc

    dset = dataloader.load_dataset(config)
    rich.print(dset)


if __name__ == "__main__":
    args = Arguments()
    run(args)
