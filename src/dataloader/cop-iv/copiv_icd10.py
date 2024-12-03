"""COP-IV-ICD10: A medical coding dataset extracted from COP-IV with ICD-10 diagnosis and procedure codes."""

from pathlib import Path
import typing as typ

import datasets
import polars as pl

from dataloader.mimic_utils import remove_rare_codes

logger = datasets.logging.get_logger(__name__)

_CITATION = """
@inproceedings{rohr_revisiting_2024,
	title = {Revisiting {Clinical} {Outcome} {Prediction} for {MIMIC}-{IV}},
	author = {Röhr, Tom and Figueroa, Alexei and Papaioannou, 
    Jens-Michalis and Fallon, Conor and Bressem, Keno and Nejdl, 
    Wolfgang and Löser, Alexander},
	year = {2024},
	publisher = {Association for Computational Linguistics},
	url = {https://aclanthology.org/2024.clinicalnlp-1.18},
	doi = {10.18653/v1/2024.clinicalnlp-1.18},
}
"""

_DESCRIPTION = """
COP-IV-ICD10: A clinical outcome prediction dataset supporting 3 tasks; patient routing, diagnoses, 
and procedure prediction. Diagnoses and procedures are derived from the relevant ICD-10 codes.
The base dataset is MIMIC-IV.
"""

_PROJECT_ROOT = Path(__file__).parent.parent
_DATASET_PATH = _PROJECT_ROOT / "data/processed/copiv.parquet"
_SPLITS_PATH = _PROJECT_ROOT / "data/splits/copiv_icd10_split.feather"


class COP_IV_ICD10_Config(datasets.BuilderConfig):
    """BuilderConfig for COP-IV-ICD10."""

    def __init__(self, **kwargs: typ.Any):
        """BuilderConfig for COP-IV-ICD10.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(COP_IV_ICD10_Config, self).__init__(**kwargs)


class COP_IV_ICD10(datasets.GeneratorBasedBuilder):
    """COP-IV-ICD10: A medical coding dataset with ICD-10 diagnosis and procedure codes."""

    BUILDER_CONFIGS = [
        COP_IV_ICD10_Config(
            name="copiv-icd9",
            version=datasets.Version("1.0.0", ""),
            description="Processed COP-IV dataset with ICD-10 codes.",
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
                    "text_admission": datasets.Value("string"),
                    "diagnosis": datasets.Sequence(datasets.Value("string")),
                    "procedure": datasets.Sequence(datasets.Value("string")),
                    "careunit": datasets.Value("string"),
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

        # drop duplicates
        data = data.unique(subset=["_id"])

        # filter out rows with no codes
        data = data.filter(pl.col("diagnosis").is_not_null() | pl.col("procedure").is_not_null())

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
