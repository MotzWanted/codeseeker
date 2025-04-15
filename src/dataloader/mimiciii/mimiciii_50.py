"""MIMIC-III-50: A public medical coding dataset from MIMIC-III with ICD-9 diagnosis and procedure codes."""

import hashlib
import json
import typing as typ

import datasets
import polars as pl

from dataloader import mimic_utils
from dataloader.constants import PROJECT_ROOT

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
MIMIC-III-50: A medical coding dataset from the Mullenbach et al. (2018) paper.
Mullenbach et al. sampled the splits randomly and didn't exclude any rare codes.
The data must not be used by or shared to someone without the license. You can obtain the license in https://physionet.org/content/mimiciii/1.4/.
"""

_MEDICAL_CODING_SYSTEMS_DIR = PROJECT_ROOT / "data/mimic-iii/raw"
_MIMICIVII_50_PATH = PROJECT_ROOT / "data/mimic-iii/processed/mimiciii_50.parquet"
_SPLITS_PATH = PROJECT_ROOT / "data/mimic-iii/splits/mimiciii_50_splits.feather"


class MIMIC_III_50_Config(datasets.BuilderConfig):
    """BuilderConfig for MIMIC-III-50."""

    def __init__(self, **kwargs: typ.Any):
        """BuilderConfig for MIMIC-III-50.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MIMIC_III_50_Config, self).__init__(**kwargs)


class MIMIC_III_50(datasets.GeneratorBasedBuilder):
    """MIMIC-III-50: A public medical coding dataset from MIMIC-III with ICD-9 diagnosis
    and procedure codes Version 1.0."""

    BUILDER_CONFIGS = [
        MIMIC_III_50_Config(
            name="mimiciii-50",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    mimic_utils.SUBJECT_ID_COLUMN: datasets.Value("int64"),
                    mimic_utils.ID_COLUMN: datasets.Value("int64"),
                    mimic_utils.ROW_ID_COLUMN: datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "codes": datasets.Sequence(datasets.Value("string")),
                    "negatives": datasets.Sequence(datasets.Value("string")),
                    "code_type": datasets.Value("string"),
                    "classes": datasets.Value("string"),
                }
            ),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:  # type: ignore
        mimiciii_50 = self._process_data()
        classes = {
            code: desc
            for codes, descriptions in zip(mimiciii_50["codes"], mimiciii_50["codes_descriptions"])
            for code, desc in zip(codes, descriptions)
        }
        if len(classes) < 50:
            raise ValueError(f"Only {len(classes)} classes available. Need at least 50 classes.")
        mimiciii_50 = mimiciii_50.with_columns([pl.lit(json.dumps(classes)).alias("classes")])
        splits = pl.read_ipc(_SPLITS_PATH)
        splits = splits.rename(
            {
                "_id": mimic_utils.ID_COLUMN,
            }
        )
        data = mimiciii_50.join(splits, on=mimic_utils.ID_COLUMN)
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

    def _get_code_descriptions(self, df: pl.DataFrame, code_column: str, code_lookup: dict[str, str]) -> pl.DataFrame:
        # Convert description DataFrame to dictionary for faster lookups
        return df.with_columns(
            pl.col(code_column)
            .map_elements(
                lambda codes: [code_lookup.get(code, None) for code in codes],
                return_dtype=pl.List(pl.Utf8),
            )
            .alias(f"{code_column}_descriptions")
        )

    def _process_data(
        self,
    ) -> pl.DataFrame:
        """Processes the raw dataset."""
        mimiciii_50 = pl.read_parquet(_MIMICIVII_50_PATH)

        code_descriptions = mimic_utils.get_icd9_descriptions(_MEDICAL_CODING_SYSTEMS_DIR)
        code_to_desc = dict(zip(code_descriptions["icd9_code"], code_descriptions["icd9_description"]))

        mimiciii_50 = self._get_code_descriptions(mimiciii_50, "codes", code_to_desc)

        return mimiciii_50.filter(pl.col("codes").is_not_null())

    def _generate_examples(  # type: ignore
        self, data: pl.DataFrame
    ) -> typ.Generator[tuple[int, dict[str, typ.Any]], None, None]:  # type: ignore
        """Generate examples from a parquet file using split information."""
        data = data.drop("split")

        # Iterate through rows and yield examples
        for row in data.to_dicts():
            _hash = hashlib.md5(json.dumps(row).encode()).hexdigest()
            yield _hash, row
