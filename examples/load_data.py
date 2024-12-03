from __future__ import annotations

import typing

import datasets
from pydantic_settings import BaseSettings, SettingsConfigDict
import rich

from dataloader import meddec, snomed, mdace_inpatient


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


DATASET_CONFIGS = {
    "meddec": {"path": meddec, "split": "train", "trust_remote_code": True},
    "snomed": {"path": snomed, "split": "train", "trust_remote_code": True},
    "mdace-icd9-diagnosis": {
        "path": mdace_inpatient,
        "name": "icd9-diagnosis",
        "split": "test",
        "trust_remote_code": True,
    },
    "mdace-icd9-procedure": {
        "path": mdace_inpatient,
        "name": "icd9-procedure",
        "split": "test",
        "trust_remote_code": True,
    },
    "mdace-icd10-diagnosis": {
        "path": mdace_inpatient,
        "name": "icd10-diagnosis",
        "split": "test",
        "trust_remote_code": True,
    },
    "mdace-icd10-procedure": {
        "path": mdace_inpatient,
        "name": "icd10-procedure",
        "split": "test",
        "trust_remote_code": True,
    },
    "my_data": {
        "identifier": "my_data",
        "path": my_loader,
    },
}


class Arguments(BaseSettings):
    """Arguments for the script."""

    name: str = "mdace-icd10-diagnosis"

    model_config = SettingsConfigDict(cli_parse_args=True, frozen=True)


def run(args: Arguments) -> None:
    """Showcase the `load_dataset` function."""
    try:
        config = DATASET_CONFIGS[args.name]
    except KeyError as exc:
        raise KeyError(f"Configuration for `{args.name}` not found!") from exc

    dset = datasets.load_dataset(**config)
    rich.print(dset)


if __name__ == "__main__":
    args = Arguments()
    run(args)
