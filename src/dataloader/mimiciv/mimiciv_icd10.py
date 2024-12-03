"""MIMIC-IV-ICD10: A medical coding dataset extracted from MIMIC-IV with ICD-10 diagnosis and procedure codes."""

from pathlib import Path
import typing as typ

import datasets
import polars as pl

from dataloader.mimic_utils import remove_rare_codes

logger = datasets.logging.get_logger(__name__)

_CITATION = """
@article{johnson2023mimiciv,
    title = {MIMIC-IV (version 2.2)},
    author = {Johnson, Alistair and Bulgarelli, Lucas and Pollard,
    Tom and Horng, Steven and Celi, Leo Anthony and Mark, Roger},
    year = {2023},
    journal = {PhysioNet},
    url = {https://doi.org/10.13026/6mm1-ek67},
    doi = {10.13026/6mm1-ek67}
}
"""

_DESCRIPTION = """
MIMIC-IV-ICD9: A medical coding dataset created from MIMIC-IV with ICD-10 diagnosis and procedure codes.
This dataset is processed to retain only relevant ICD-10 codes, filter rare codes, and ensure no duplicate entries.
It also includes train/validation/test splits.
"""

_PROJECT_ROOT = Path(__file__).parent.parent
_DATASET_PATH = _PROJECT_ROOT / "data/processed/mimiciv.parquet"
_SPLITS_PATH = _PROJECT_ROOT / "data/splits/mimiciv_icd10_split.feather"


class MIMIC_IV_ICD10_Config(datasets.BuilderConfig):
    """BuilderConfig for MIMIC-IV-ICD9."""

    def __init__(self, **kwargs: typ.Any):
        """BuilderConfig for MIMIC-IV-ICD9.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MIMIC_IV_ICD10_Config, self).__init__(**kwargs)


class MIMIC_IV_ICD10(datasets.GeneratorBasedBuilder):
    """MIMIC-IV-ICD10: A medical coding dataset with ICD-10 diagnosis and procedure codes."""

    BUILDER_CONFIGS = [
        MIMIC_IV_ICD10_Config(
            name="mimiciv-icd9",
            version=datasets.Version("1.0.0", ""),
            description="Processed MIMIC-IV dataset with ICD-9 codes.",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "_id": datasets.Value("int64"),
                    "subject_id": datasets.Value("int64"),
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

        # Load the main dataset and split information
        splits = pl.read_ipc(splits_path)
        raw_data = pl.read_parquet(data_path)
        data = raw_data.join(splits, on="_id")

        # remove not used columns
        data = data.drop(["note_seq", "charttime", "storetime"])

        # only keep ICD-10 codes
        data = data.filter((pl.col("diagnosis_code_type") == "icd10cm") | (pl.col("procedure_code_type") == "icd10pcs"))

        # remove rare codes
        data = remove_rare_codes(data, ["diagnosis_codes", "procedure_codes"], 10)

        # drop duplicates
        data = data.unique(subset=["_id"])

        # filter out rows with no codes
        data = data.filter(pl.col("diagnosis_codes").is_not_null() | pl.col("procedure_codes").is_not_null())

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

    def _generate_examples(  # type: ignore
        self, data: pl.DataFrame
    ) -> typ.Generator[tuple[int, dict[str, typ.Any]], None, None]:  # type: ignore
        """Generate examples from a parquet file using split information."""
        data = data.drop("split")

        # Iterate through rows and yield examples
        for row in data.to_dicts():
            yield row["_id"], row
