from __future__ import annotations
import typing

import datasets
import rich
from pydantic_settings import BaseSettings, SettingsConfigDict
import statistics

import dataloader
from dataloader.tanner.constants import (
    TANNER_ED_PATH,
    TANNER_INPATIENT_PATH,
    TANNER_SURGERY_PATH,
)
from dataloader.base import DatasetConfig


DATASET_CONFIGS = {
    "tanner-health-ed": {
        "identifier": "tanner-health-ed",
        "name_or_path": TANNER_ED_PATH,
        "split": "test",
        "options": {
            "adapter": "TannerAdapter",
        },
        "kwargs": {
            "formatting": "newline",
        },
    },
    "tanner-health-inpatient": {
        "identifier": "tanner-health-inpatient",
        "name_or_path": TANNER_INPATIENT_PATH,
        "split": "test",
        "options": {
            "adapter": "TannerAdapter",
        },
        "kwargs": {
            "formatting": "newline",
        },
    },
    "tanner-health-surgery": {
        "identifier": "tanner-health-surgery",
        "name_or_path": TANNER_SURGERY_PATH,
        "split": "test",
        "options": {
            "adapter": "TannerAdapter",
        },
        "kwargs": {
            "formatting": "newline",
        },
    },
}


class Arguments(BaseSettings):
    """Arguments for the script."""

    name: str = "tanner-health-ed"

    model_config = SettingsConfigDict(cli_parse_args=True, frozen=True)


def calculate_statistics(
    dset: datasets.Dataset | datasets.DatasetDict,
) -> dict[str, typing.Any]:
    """Calculate descriptive statistics for the dataset."""
    # Extract data
    if isinstance(dset, datasets.DatasetDict):
        return {split: calculate_statistics(dset[split]) for split in dset.keys()}
    target_counts = [len(row) for row in dset["targets"]]
    note_lengths = [len(row) for row in dset["note"]]
    all_targets = [target for row in dset["targets"] for target in row]

    return {
        "unique_target_codes": len(set(all_targets)),
        "target_codes_per_row": {
            "min": min(target_counts),
            "max": max(target_counts),
            "average": round(statistics.mean(target_counts), 2),
        },
        "note_lengths": {
            "min": min(note_lengths),
            "max": max(note_lengths),
            "average": round(statistics.mean(note_lengths), 2),
        },
        "total_rows": len(dset),
    }


def run(args: Arguments) -> None:
    """Showcase the `load_dataset` function."""
    try:
        config = DatasetConfig(**DATASET_CONFIGS[args.name])
    except KeyError as exc:
        raise KeyError(f"Configuration for `{args.name}` not found!") from exc
    config.options.prep_map_kws = {"num_proc": 1, "load_from_cache_file": False}
    dset = dataloader.load_dataset(config)
    dset = dset.filter(lambda x: x["claim_denial"] == "APPROVED")

    # Display dataset info
    rich.print(f"\n[bold blue]Dataset: {args.name}[/bold blue]")
    rich.print(dset)

    # Calculate and display statistics
    stats = calculate_statistics(dset)

    rich.print("\n[bold green]Descriptive Statistics:[/bold green]")
    rich.print(f"Total rows: {stats['total_rows']}")
    rich.print(f"Number of unique target codes: {stats['unique_target_codes']}")

    rich.print("\n[bold yellow]Target codes per row:[/bold yellow]")
    rich.print(stats["target_codes_per_row"])

    rich.print("\n[bold yellow]Note lengths (characters):[/bold yellow]")
    rich.print(stats["note_lengths"])


if __name__ == "__main__":
    args = Arguments()
    run(args)
