from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

import rich

from dataloader.loaders import nbme_notes


class Arguments(BaseSettings):
    """Args for the script."""

    path: str = str(Path("~/Downloads/nbme-score-clinical-patient-notes/").expanduser())
    num_workers: int = 1
    batch_size: int = 1
    n_samples: int = 300
    fewshot: int = 5
    seed: int = 42

    model_config = SettingsConfigDict(cli_parse_args=True)


def run(args: Arguments):
    dataset = nbme_notes.NbmeDatasetLoader().load_dataset(size=300, n_shots=5)

    rich.print(dataset, dataset["train"][0], sep="\n\n")


if __name__ == "__main__":
    args = Arguments()
    run(args)
