from __future__ import annotations

import json
from pathlib import Path
import typing

import datasets
import rich
from pydantic_settings import BaseSettings, SettingsConfigDict

import dataloader
from dataloader import mdace_inpatient, meddec, mimiciii_50, mimiciv, mimiciv_50, snomed
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
        "subsets": ["icd10"],
        "options": {"segmenter": SEGMENTER, "adapter": "MimicIdentifyAdapter"},
    },
    "my_data": {
        "identifier": "my_data",
        "name_or_path": my_loader,
        "path": my_loader,
    },
    "mimic-iii-50": {
        "identifier": "mimic-iii-50",
        "name_or_path": mimiciii_50,
        "options": {"segmenter": SEGMENTER, "adapter": "MimicForTrainingAdapter"},
    },
    "mimic-iv-50": {
        "identifier": "mimic-iv-50",
        "name_or_path": mimiciv_50,
        "subsets": ["icd10"],
        "options": {"segmenter": SEGMENTER, "adapter": "MimicForTrainingAdapter"},
    },
}


class Arguments(BaseSettings):
    """Arguments for the script."""

    name: str = "mimic-iv"

    model_config = SettingsConfigDict(cli_parse_args=True, frozen=True)


keywords = ["neonatal", "apgar", "caesarean", "labour"]
# keywords = ["apgar"]


def run(args: Arguments) -> None:
    """Showcase the `load_dataset` function."""
    try:
        config = DatasetConfig(**DATASET_CONFIGS[args.name])
    except KeyError as exc:
        raise KeyError(f"Configuration for `{args.name}` not found!") from exc
    config.options.prep_map_kws = {"num_proc": 16, "load_from_cache_file": False}
    dset = dataloader.load_dataset(config)
    obstetrics_dset = dset.filter(lambda row: any(c[:2] == "10" or c[0] == "O" for c in row["codes"]))
    obstetrics_dset = obstetrics_dset.filter(lambda row: any(kw in row["text"].lower() for kw in keywords))
    # obstetrics_service_dset = dset.filter(lambda row: "\nService: OBSTETRICS/GYNECOLOGY\n" in row["text"])
    rich.print(obstetrics_dset)
    # rich.print(obstetrics_service_dset)
    # intersection = obstetrics_service_dset.filter(
    #     lambda row: not any(c[:2] == "10" or c[0] == "O" for c in row["codes"])
    # )
    # save each row to a json file
    # output_dir_1 = Path("~/Downloads/obstetrics_service").expanduser()
    # output_dir_1.mkdir(exist_ok=True, parents=True)
    # for _, split in obstetrics_dset.items():
    #     for i, row in enumerate(split):
    #         file_path = output_dir_1 / f"discharge_summary_{i}.json"
    #         with open(file_path, "w", encoding="utf-8") as f:
    #             json.dump(row, f, indent=2)  # Corrected line
    output_dir_2 = Path("~/Downloads/obstetric_keywords").expanduser()
    output_dir_2.mkdir(exist_ok=True, parents=True)
    for _, split in obstetrics_dset.items():
        for i, row in enumerate(split):
            file_path = output_dir_2 / f"discharge_summary_{i}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)  # Corrected line


if __name__ == "__main__":
    args = Arguments()
    run(args)
