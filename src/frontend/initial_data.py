import datasets
from dataloader import nmbe
import dataloader
from dataloader.base import DatasetConfig
from segmenters import factory


def _get_dataset(dset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    """Get a `datasets.Dataset`."""
    if isinstance(dset, datasets.Dataset):
        return dset
    return next(iter(dset.values()))


config = DatasetConfig(
    identifier="nbme",
    name_or_path=nmbe,
    split="test",
    options={"segmenter": factory("nbme", spacy_model="en_core_web_lg")},
)

nbme_data = dataloader.load_dataset(config)
entry_count = len(nbme_data)
