"""MIMIC-III-50: A public medical coding dataset from MIMIC-III with ICD-9 diagnosis and procedure codes."""

from pathlib import Path
import typing as typ

import datasets
import polars as pl

from dataloader import mimic_utils
from src.dataloader.mimic_utils import keep_top_k_codes

logger = datasets.logging.get_logger(__name__)

_CITATION = """
@inproceedings{mullenbach-etal-2018-explainable,
    title = "Explainable Prediction of Medical Codes from Clinical Text",
    author = "Mullenbach, James  and
      Wiegreffe, Sarah  and
      Duke, Jon  and
      Sun, Jimeng  and
      Eisenstein, Jacob",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N18-1100",
    doi = "10.18653/v1/N18-1100",
    pages = "1101--1111",
}
"""

_DESCRIPTION = """
MIMIC-III-50: A medical coding dataset created from MIMIC-III with ICD-9 diagnosis and procedure codes.
This dataset includes the top 50 most common codes and ensures no duplicate entries.
It is split into train/validation/test based on a provided split file.
"""

_PROJECT_ROOT = Path(__file__).parent.parent
_DATASET_PATH = _PROJECT_ROOT / "data/processed/mimiciii.parquet"
_SPLITS_PATH = _PROJECT_ROOT / "data/splits/mimiciii_50_splits.feather"


class MIMIC_III_50_Config(datasets.BuilderConfig):
    """BuilderConfig for MIMIC-III-50."""

    def __init__(self, **kwargs: typ.Any):
        """BuilderConfig for MIMIC-III-50.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MIMIC_III_50_Config, self).__init__(**kwargs)


class MIMIC_III_50(datasets.GeneratorBasedBuilder):
    """MIMIC-III-50: A medical coding dataset from MIMIC-III with ICD-9 diagnosis and procedure codes."""

    BUILDER_CONFIGS = [
        MIMIC_III_50_Config(
            name="mimiciii-50",
            version=datasets.Version("1.0.0", ""),
            description="Top 50 ICD-9 codes dataset from MIMIC-III.",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    mimic_utils.SUBJECT_ID_COLUMN: datasets.Value("int64"),
                    mimic_utils.ID_COLUMN: datasets.Value("int64"),
                    "note_type": datasets.Value("string"),
                    "note_subtype": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "diagnosis_codes": datasets.Sequence(datasets.Value("string")),
                    "diagnosis_code_type": datasets.Value("string"),
                    "procedure_codes": datasets.Sequence(datasets.Value("string")),
                    "procedure_code_type": datasets.Value("string"),
                }
            ),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:  # type: ignore
        data_path = dl_manager.download_and_extract(str(_DATASET_PATH))
        splits_path = dl_manager.download_and_extract(str(_SPLITS_PATH))

        # Load raw data and split information
        splits = pl.read_ipc(splits_path)
        raw_data = pl.read_parquet(data_path)
        data = raw_data.join(splits, on=mimic_utils.ID_COLUMN)

        # Process the dataset
        data = self._process_data(data)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data": data.filter(pl.col("split") == "train")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data": data.filter(pl.col("split") == "val")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data": data.filter(pl.col("split") == "test")},
            ),
        ]

    def _process_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """Process the raw dataset."""
        # Remove unused columns
        data = data.drop(["note_seq", "charttime", "storetime"])

        # Keep only ICD-9 codes
        data = data.filter((pl.col("diagnosis_code_type") == "icd9cm") | (pl.col("procedure_code_type") == "icd9pcs"))

        # Remove duplicates
        data = data.unique(subset=[mimic_utils.ID_COLUMN])

        # Keep only the top 50 codes
        data = keep_top_k_codes(data, ["diagnosis_codes", "procedure_codes"], 50)

        # Filter out rows without codes
        data = data.filter(pl.col("diagnosis_codes").is_not_null() | pl.col("procedure_codes").is_not_null())

        return data

    def _generate_examples(  # type: ignore
        self, data: pl.DataFrame
    ) -> typ.Generator[tuple[int, dict[str, typ.Any]], None, None]:  # type: ignore
        """Generate examples from the dataset."""
        data = data.drop("split")

        for row in data.to_dicts():
            yield row[mimic_utils.ID_COLUMN], row
