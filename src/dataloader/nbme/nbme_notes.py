from collections import defaultdict
import random
import typing as typ

import datasets
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
import numpy as np

from dataloader.nbme.constants import NBME_PATH

logger = datasets.logging.get_logger(__name__)


class NbmeDatasetLoader:
    """A dataset loader for the NBME clinical patient note dataset."""

    def __init__(self, seed: int = 42):
        """Initialize the dataset loader."""
        self.seed = seed

    def __call__(
        self,
        subset: None | str = None,
        split: None | str = None,
        size: int | None = None,
        in_domain_shots: bool = True,
        **kws: typ.Any,
    ) -> datasets.Dataset | datasets.DatasetDict:
        """Load the dataset."""
        data: datasets.Dataset | datasets.DatasetDict = datasets.load_dataset(NBME_PATH, trust_remote_code=True, **kws)  # type: ignore
        logger.info(f"Loaded dataset: {data.__class__.__name__}")
        disable_progress_bar()
        logger.info(f"Extracting data: subset={subset}, split={split}, size={size}")
        data = self._extract_fewshots(data, in_domain_shots)
        if split and isinstance(data, datasets.DatasetDict):
            data = data[split]
        if size:
            data = self._extract_subset(data, size)
        enable_progress_bar()
        return data

    def _extract_subset(
        self, dset: datasets.Dataset | datasets.DatasetDict, size: int
    ) -> datasets.Dataset | datasets.DatasetDict:
        """Take a subset of the dataset."""
        if isinstance(dset, datasets.Dataset):
            return self._take_subset(dset, size, self.seed)
        return datasets.DatasetDict({k: self._take_subset(v, size, self.seed) for k, v in dset.items()})

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
                lambda row: {"fewshots": fewshot_data[row["case_num"]]},
            )
            return data
        for split, dset in data.items():
            fewshot_data = self._get_fewshot_data(dset, shots_from_same_patient)
            data[split] = dset.map(
                lambda row: {"fewshots": fewshot_data[row["case_num"]]},
            )

        return data

    def _get_fewshot_data(self, data: datasets.Dataset, shots_from_same_patient: bool) -> defaultdict:
        """Extract fewshot data with optimized lookup."""
        # Create a mapping from case_num to list of rows
        case_to_rows = defaultdict(list)
        for idx in range(len(data)):
            row = data[idx]
            case_to_rows[row["case_num"]].append(row)

        fewshot_data = defaultdict(list)
        for case_num, rows in case_to_rows.items():
            if shots_from_same_patient:
                fewshot_data[case_num] = self._get_valid_rows(rows)
            else:
                fewshot_data[case_num] = self._get_valid_rows_from_other_cases(case_num, case_to_rows)

        return fewshot_data

    def _get_valid_rows_from_other_cases(self, current_case_num: int, case_to_rows: defaultdict) -> list:
        """Get valid rows from other cases excluding the current case."""
        valid_rows = []
        rows_per_case = int(len(case_to_rows[current_case_num]) / len(case_to_rows))
        for other_case_num, other_rows in case_to_rows.items():
            random.seed(self.seed)
            sampled_rows = random.sample(other_rows, min(rows_per_case, len(other_rows)))
            if other_case_num != current_case_num:
                valid_rows.extend(
                    self._get_valid_rows(sampled_rows),
                )
        return valid_rows

    @staticmethod
    def _get_valid_rows(rows: list) -> list:
        """Get valid rows with labels."""
        return [row for row in rows if isinstance(row["labels"], dict) and len(row["labels"]) > 0]
