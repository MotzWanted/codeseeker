from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

import matplotlib.pyplot as plt
import datasets

from dataloader.adapters.alignment import NmbeAdapter
from dataloader.loaders.nbme_notes import NbmeDatasetLoader
from segmenters import factory, Segmenter

PLOTS_FOLDER = Path(__file__).parent.parent / "reports" / "plots"


class Arguments(BaseSettings):
    """Args for the script."""

    split: str = "test"
    segmenter: str = "nbme:window:spacy"  # spacy | sentence | nbme
    spacy_model: str = "en_core_web_trf"
    n_samples: int = 500

    corpus_key: str = "features"  # the corpus indices to predict
    query_key: str = "patient_note"  # the queries given for predicting indices, e.g., transcript or patient note

    num_workers: int = 12

    model_config = SettingsConfigDict(cli_parse_args=True)


def _get_dataset(dset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    """Get a `datasets.Dataset`."""
    if isinstance(dset, datasets.Dataset):
        return dset
    return next(iter(dset.values()))


def plot_query_lengths(segmented_data: list[datasets.Dataset], segmenters: list[str]):
    fig, ax = plt.subplots(figsize=(10, 6))

    query_lengths_list = [[len(query) for queries in data["queries"] for query in queries] for data in segmented_data]

    for lengths, segmenter in zip(query_lengths_list, segmenters):
        ax.hist(lengths, bins=10, alpha=0.5, label=segmenter)

    ax.set_title("Distribution of Query Lengths Across Segmenters")
    ax.set_xlabel("Length of Queries")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "query_lengths_histogram.png")


def plot_number_of_queries(segmented_data: list[datasets.Dataset], segmenters: list[str]):
    fig, ax = plt.subplots(figsize=(10, 6))

    number_of_queries_list = [[len(queries) for queries in data["queries"]] for data in segmented_data]

    for lengths, segmenter in zip(number_of_queries_list, segmenters):
        ax.hist(lengths, bins=15, alpha=0.5, label=segmenter)

    ax.set_title("Distribution of Number of Queries Across Segmenters")
    ax.set_xlabel("Number of Queries")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "number_of_queries_histogram.png")


def plot_labels_lengths(segmented_data: list[datasets.Dataset], segmenters: list[str]):
    fig, ax = plt.subplots(figsize=(10, 6))

    labels_lengths_list = [[len(x) for sublist in data["labels"] for x in sublist] for data in segmented_data]

    for lengths, segmenter in zip(labels_lengths_list, segmenters):
        ax.hist(lengths, bins=5, alpha=0.5, label=segmenter)

    ax.set_title("Distribution of Corpus Lengths Across Segmenters")
    ax.set_xlabel("Length of Corpus Entries")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER / "labels_lengths_histogram.png")


def run(args: Arguments):
    """Run the script."""
    data: datasets.Dataset = NbmeDatasetLoader().load_dataset(split="test", size=args.n_samples)
    segmenters_ = [x for x in args.segmenter.split(":")] if ":" in args.segmenter else [args.segmenter]
    segmented_data = []
    for s in segmenters_:
        segmenter: Segmenter = factory(s, args.spacy_model)  # noqa: F821
        adapter = NmbeAdapter(segmenter=segmenter, query_key=args.query_key)
        temp_data = data.map(
            adapter,
            num_proc=args.num_workers,
            desc=f"Adapting dataset to `{NmbeAdapter.__name__}` using `{segmenter.__class__.__name__}`.",
            remove_columns=_get_dataset(data).column_names,
        )
        segmented_data.append(temp_data)
    plot_query_lengths(segmented_data, segmenters_)
    plot_number_of_queries(segmented_data, segmenters_)
    plot_labels_lengths(segmented_data, segmenters_)


if __name__ == "__main__":
    args = Arguments()
    run(args)
