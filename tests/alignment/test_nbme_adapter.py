from pathlib import Path
import datasets
import pytest

from dataloader.adapt.alignment import NbmeAdapter
from segmenters.base import Segmenter, factory

BASE_PATH = Path("tests/alignment/data")
DATASET_FILE_NAME = "nbme_subset.json"


@pytest.fixture
def dataset() -> datasets.Dataset:
    dset = datasets.load_dataset("json", data_files=str(BASE_PATH / DATASET_FILE_NAME))
    return dset["train"] if isinstance(dset, datasets.DatasetDict) else dset


@pytest.fixture
def segmenter() -> Segmenter:
    return factory("nbme", "en_core_web_sm")


def _get_dataset(dset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    """Get a `datasets.Dataset`."""
    if isinstance(dset, datasets.Dataset):
        return dset
    return next(iter(dset.values()))


@pytest.mark.parametrize("seeds", [[42, 43, 44], [1, 2]])
def test_shuffling_of_target(dataset: datasets.Dataset, segmenter: Segmenter, seeds: list[int]) -> None:
    """Test that the target is shuffled yielding different targets for each seed."""
    adapted_targets = []
    for seed in seeds:
        adapter = NbmeAdapter(segmenter=segmenter, query_key="patient_note", seed=seed)
        adapted_dset = dataset.map(
            adapter,
            num_proc=1,
            desc=f"Adapting dataset to `AlignmentModel` using `{NbmeAdapter.__name__}`.",
            remove_columns=_get_dataset(dataset).column_names,
            load_from_cache_file=False,
        )
        adapted_targets.append(adapted_dset["labels"])

    # Assert that each target sequence has the same length across seeds
    for i in range(1, len(adapted_targets)):
        assert len(adapted_targets[0]) == len(adapted_targets[i]), "Target lengths differ across seeds."

    # Compare each target element pairwise across seeds to verify they are shuffled
    num_elements = len(adapted_targets[0])
    for i in range(num_elements):
        for j in range(1, len(adapted_targets)):
            assert (
                adapted_targets[0][i] != adapted_targets[j][i]
            ), f"Target element {i} is not shuffled with seed {seeds[j]}."
            assert len(adapted_targets[0][i]) == len(adapted_targets[j][i]), "Shuffled target length mismatch."
