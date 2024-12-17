"""MedDec: Medical Decisions for Discharge Summaries in the MIMIC-III Database."""

import json
from pathlib import Path
import re
import typing as typ

import datasets
import polars as pl

from dataloader import mimic_utils
from dataloader.constants import PROJECT_ROOT

logger = datasets.logging.get_logger(__name__)

_CITATION = """
@article{elgaar2024meddec,
    title = {MedDec: Medical Decisions for Discharge Summaries in the MIMIC-III Database (version 1.0.0)},
    author = {Elgaar, M. and Cheng, J. and Vakil, N. and Amiri, H. and Celi, L. A.},
    year = {2024},
    journal = {PhysioNet},
    url = {https://doi.org/10.13026/nqnw-7d62},
    doi = {10.13026/nqnw-7d62}
}
"""

_DESCRIPTION = """
MedDec: A dataset containing medical decisions annotated in discharge summaries from the MIMIC-III database.
Annotations include decisions, categories, and offset spans,
supporting tasks like evidence extraction and clinical decision modeling.
The dataset is split into train and validation sets.
"""

_DATA_PATH = PROJECT_ROOT / "data/meddec/processed"
_SPLIT_PATH = PROJECT_ROOT / "data/meddec/splits"


class MedDecConfig(datasets.BuilderConfig):
    """BuilderConfig for MedDec."""

    def __init__(self, **kwargs: typ.Any):
        """BuilderConfig for MedDec.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MedDecConfig, self).__init__(**kwargs)


class MedDec(datasets.GeneratorBasedBuilder):
    """MedDec: Medical Decisions Dataset."""

    BUILDER_CONFIGS = [
        MedDecConfig(
            name="meddec",
            version=datasets.Version("1.0.0", ""),
            description="Processed MedDec dataset for medical decisions in discharge summaries.",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    mimic_utils.SUBJECT_ID_COLUMN: datasets.Value("int64"),
                    mimic_utils.ID_COLUMN: datasets.Value("int64"),
                    mimic_utils.ROW_ID_COLUMN: datasets.Value("int64"),
                    "file_name": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "annotations": datasets.Sequence(
                        {
                            "annotator_id": datasets.Value("string"),
                            "start_offset": datasets.Value("int64"),
                            "end_offset": datasets.Value("int64"),
                            "category": datasets.Value("string"),
                            "category_id": datasets.Value("string"),
                            "decision": datasets.Value("string"),
                            "annotation_id": datasets.Value("string"),
                        }
                    ),
                    "classes": datasets.Value("string"),
                    "fewshots": datasets.Sequence(
                        {
                            "annotator_id": datasets.Value("string"),
                            "start_offset": datasets.Value("int64"),
                            "end_offset": datasets.Value("int64"),
                            "category": datasets.Value("string"),
                            "category_id": datasets.Value("string"),
                            "decision": datasets.Value("string"),
                            "annotation_id": datasets.Value("string"),
                        }
                    ),
                }
            ),
            citation=_CITATION,
        )

    def aggregate_rows(self, data: pl.DataFrame) -> pl.DataFrame:
        """Process the MedDec data."""
        return data.group_by(
            [mimic_utils.SUBJECT_ID_COLUMN, mimic_utils.ID_COLUMN, mimic_utils.ROW_ID_COLUMN, "file_name", "text"]
        ).agg(
            [
                # Collect all annotations as a list of dictionaries
                pl.struct(
                    [
                        "annotator_id",
                        "start_offset",
                        "end_offset",
                        "category",
                        "category_id",
                        "decision",
                        "annotation_id",
                    ]
                ).alias("annotations")
            ]
        )

    @staticmethod
    def split_category(category: str) -> tuple[int, str]:
        pattern = r"Category (\d+):\s*(.*)"
        match = re.match(pattern, category)
        if match:
            return match.group(1), match.group(2)
        else:
            return 0, "Unknown"

    def _split_generators(  # type: ignore
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:  # type: ignore
        train_files = self._load_split_files(_SPLIT_PATH / "train.txt")
        val_files = self._load_split_files(_SPLIT_PATH / "val.txt")

        data = pl.read_parquet(_DATA_PATH / "meddec.parquet")
        filtered_data = data.with_columns(
            pl.col("category").str.extract(r"Category (\d+):", 1).cast(pl.Int32).alias("category_id"),
            pl.col("category").str.extract(r"Category \d+: (.*)", 1).alias("category"),
        ).filter((pl.col("category_id").is_not_null()) & (pl.col("category").is_not_null()))

        classes = {
            idx: value for idx, value in zip(filtered_data["category_id"], filtered_data["category"]) if idx and value
        }
        aggregated_data = self.aggregate_rows(filtered_data)
        aggregated_data = aggregated_data.with_columns([pl.lit(json.dumps(classes)).alias("classes")])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data": aggregated_data.filter(pl.col("file_name").is_in(train_files))},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data": aggregated_data.filter(pl.col("file_name").is_in(val_files))},
            ),
        ]

    def _merge_fewshot_data(self, data: pl.DataFrame, fewshot_data: pl.DataFrame) -> pl.DataFrame:
        """Get fewshot data."""

        # Perform an inner join to combine rows based on keys with suffixed column names
        joined_data = data.lazy().join(
            fewshot_data.lazy(),
            how="inner",
            left_on=["annotator_id", "annotation_id"],
            right_on=["annotator_id", "annotation_id"],
            suffix="_right",
        )

        # Exclude rows where annotator_id and annotation_id match
        filtered_data = (
            joined_data.filter(
                (pl.col("annotator_id") != pl.col("annotator_id_right"))
                | (pl.col("annotation_id") != pl.col("annotation_id_right"))
            )
            .group_by(["annotator_id", "annotation_id"])
            .agg(
                pl.struct(
                    [
                        "annotator_id_right",
                        "annotation_id_right",
                        "start_offset",
                        "end_offset",
                        "category",
                        "category_id",
                        "decision",
                    ]
                ).alias("fewshots")
            )
        )

        # Collect the filtered data
        filtered_data_collected = filtered_data.collect()

        # Join the fewshots column back to the original dataset
        data = data.join(filtered_data_collected, on=["annotator_id", "annotation_id"], how="left")

        return data

    def _load_split_files(self, filepath: Path) -> list[str]:
        """Load the list of file names for a given split."""
        with filepath.open("r") as f:
            return [line.strip() for line in f.readlines()]

    def _generate_examples(  # type: ignore
        self, data: pl.DataFrame
    ) -> typ.Generator[tuple[int, dict[str, typ.Any]], None, None]:  # type: ignore
        """Generate examples from the dataset."""
        for row in data.to_dicts():
            file_name = row.pop("file_name")
            _hash = hash(file_name)
            yield _hash, row
