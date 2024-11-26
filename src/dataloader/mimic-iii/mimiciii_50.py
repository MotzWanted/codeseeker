"""MIMIC-III-50: A public medical coding dataset from MIMIC-III with ICD-9 diagnosis and procedure codes."""

from pathlib import Path
import typing as typ

import datasets
import polars as pl

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

PROJECT_ROOT = Path(__file__).parent.parent.parent
_URL = PROJECT_ROOT / "data/mimic-iii/processed/mimiciii_50/"
_URLS = {
    "train": _URL / "train.parquet",
    "val": _URL / "val.parquet",
    "test": _URL / "test.parquet",
}


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
                    "subject_id": datasets.Value("int64"),
                    "_id": datasets.Value("int64"),
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
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=str(datasets.Split.TRAIN),
                gen_kwargs={"filepath": downloaded_files["train"]},  # type: ignore
            ),
            datasets.SplitGenerator(
                name=str(datasets.Split.VALIDATION),
                gen_kwargs={"filepath": downloaded_files["val"]},  # type: ignore
            ),
            datasets.SplitGenerator(
                name=str(datasets.Split.TEST),
                gen_kwargs={"filepath": downloaded_files["test"]},  # type: ignore
            ),
        ]

    def _generate_examples(self, filepath: str) -> typ.Generator[tuple[int, dict[str, typ.Any]], None, None]:  # type: ignore
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        dataframe = pl.read_parquet(filepath)

        for row in dataframe.to_dicts():
            yield row["_id"], row
            key += 1
