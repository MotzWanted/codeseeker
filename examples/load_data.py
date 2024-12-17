from __future__ import annotations

import typing

import datasets
from pydantic_settings import BaseSettings, SettingsConfigDict
import rich

from dataloader import meddec, snomed, mdace_inpatient
import dataloader
from dataloader.base import DatasetConfig
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


SEGMENTER = factory("spacy", spacy_model="en_core_web_lg")

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
    "mdace-icd10-3": {
        "identifier": "mdace-icd10",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10-diagnosis", "icd10-procedure"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "my_data": {
        "identifier": "my_data",
        "path": my_loader,
    },
}


class Arguments(BaseSettings):
    """Arguments for the script."""

    name: str = "mdace-diagnosis"

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
