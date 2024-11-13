from collections import defaultdict
import pathlib
import typing as typ

import datasets
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
import numpy as np

from dataloader.loaders.generators import nbme_kaggle_dset

logger = datasets.logging.get_logger(__name__)

LabelsKey = typ.Literal["patient_note", "rubric_features"]


class NbmeDatasetLoader:
    """A dataset loader for the NBME clinical patient note dataset."""

    def __call__(
        self,
        subset: None | str = None,
        split: None | str = None,
        size: int | None = None,
        shots_from_same_patient: bool = True,
        seed: int = 42,
        num_proc: int = 1,
        **kws: typ.Any,
    ) -> datasets.Dataset | datasets.DatasetDict:
        """Load the dataset."""
        nbme_notes_path = pathlib.Path(nbme_kaggle_dset.__file__)
        data: datasets.Dataset | datasets.DatasetDict = datasets.load_dataset(
            nbme_notes_path.as_posix(), trust_remote_code=True, **kws
        )  # type: ignore
        logger.info(f"Loaded dataset: {data.__class__.__name__}")
        disable_progress_bar()
        logger.info(f"Extracting data: subset={subset}, split={split}, size={size}")
        data = self._extract_fewshots(data, shots_from_same_patient)
        if split and isinstance(data, datasets.DatasetDict):
            data = data[split]
        if size:
            data = self._extract_subset(data, size, seed, num_proc)
        enable_progress_bar()
        return data

    def _extract_subset(
        self, dset: datasets.Dataset | datasets.DatasetDict, size: int, seed: int, num_proc: int
    ) -> datasets.Dataset | datasets.DatasetDict:
        """Take a subset of the dataset."""
        if isinstance(dset, datasets.Dataset):
            return self._take_subset(dset, size, seed)
        return datasets.DatasetDict({k: self._take_subset(v, size, seed) for k, v in dset.items()})

    @staticmethod
    def _take_subset(data: datasets.Dataset, size: int, seed: int) -> datasets.Dataset:
        """Get a subset of the data where each case number is equally represented."""
        case_numbers = data.unique("case_num")
        num_cases = len(case_numbers)
        samples_per_case = max(1, size // num_cases)

        rgn = np.random.RandomState(seed)

        # Build a mapping from case_num to list of indices
        case_to_indices = defaultdict(list)
        for idx, case_num in enumerate(data["case_num"]):
            case_to_indices[case_num].append(idx)

        sampled_indices = []
        for case in case_numbers:
            indices = case_to_indices[case]
            # Ensure we don't sample more than available indices
            actual_samples = min(samples_per_case, len(indices))
            selected_indices = rgn.choice(indices, size=actual_samples, replace=False)
            sampled_indices.extend(selected_indices)

        return data.select(sampled_indices)

    def _extract_fewshots(
        self, data: datasets.Dataset | datasets.DatasetDict, shots_from_same_patient: bool
    ) -> datasets.Dataset | datasets.DatasetDict:
        """Extract fewshot data."""
        if isinstance(data, datasets.Dataset):
            fewshot_data = self._get_fewshot_data(data, shots_from_same_patient)
            data = data.map(
                lambda row: {"few_shot": fewshot_data[row["case_num"]]},
            )
            return data
        for split, dset in data.items():
            fewshot_data = self._get_fewshot_data(dset, shots_from_same_patient)
            data[split] = dset.map(
                lambda row: {"few_shot": fewshot_data[row["case_num"]]},
            )

        return data

    def _get_fewshot_data(self, data: datasets.Dataset, shots_from_same_patient: bool) -> defaultdict:
        """Extract fewshot data."""
        case_numbers = data.unique("case_num")

        fewshot_data = defaultdict(list)
        for case in case_numbers:
            if shots_from_same_patient:
                case_subset = data.filter(lambda x: x["case_num"] == case)
            else:
                case_subset = data.filter(lambda x: x["case_num"] != case)
            case_fewshots = case_subset.filter(lambda x: isinstance(x["labels"], dict) and len(x["labels"]) > 0)
            fewshot_data[case] = case_fewshots.to_list()

        return fewshot_data
