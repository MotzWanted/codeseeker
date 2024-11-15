import datasets
from dataloader.loaders.nbme.nbme_notes import NbmeDatasetLoader
from dataloader.adapters.alignment import NbmeAdapter
from segmenters import factory


def _get_dataset(dset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    """Get a `datasets.Dataset`."""
    if isinstance(dset, datasets.Dataset):
        return dset
    return next(iter(dset.values()))


nbme_data: datasets.Dataset = NbmeDatasetLoader().load_dataset(split="test")  # type: ignore

adapter = NbmeAdapter(segmenter=factory("nbme", spacy_model="en_core_web_lg"))
nbme_data = nbme_data.map(
    adapter,
    desc=f"Adapting dataset to `AlignmentModel` using `{NbmeAdapter.__name__}`.",
    remove_columns=_get_dataset(nbme_data).column_names,
)
entry_count = len(nbme_data)
