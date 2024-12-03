"""MedDec: Medical Decisions for Discharge Summaries in the MIMIC-III Database."""

from pathlib import Path
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
                    "annotator_id": datasets.Value("string"),
                    "file_name": datasets.Value("string"),
                    "start_offset": datasets.Value("int64"),
                    "end_offset": datasets.Value("int64"),
                    "category": datasets.Value("string"),
                    "decision": datasets.Value("string"),
                    "annotation_id": datasets.Value("string"),
                    mimic_utils.SUBJECT_ID_COLUMN: datasets.Value("int64"),
                    mimic_utils.ID_COLUMN: datasets.Value("int64"),
                    mimic_utils.ROW_ID_COLUMN: datasets.Value("int64"),
                    "text": datasets.Value("string"),
                }
            ),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:  # type: ignore
        train_files = self._load_split_files(_SPLIT_PATH / "train.txt")
        val_files = self._load_split_files(_SPLIT_PATH / "val.txt")

        data = pl.read_parquet(_DATA_PATH / "meddec.parquet")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data": data.filter(pl.col("file_name").is_in(train_files))},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data": data.filter(pl.col("file_name").is_in(val_files))},
            ),
        ]

    def _load_split_files(self, filepath: Path) -> list[str]:
        """Load the list of file names for a given split."""
        with filepath.open("r") as f:
            return [line.strip() for line in f.readlines()]

    def _generate_examples(  # type: ignore
        self, data: pl.DataFrame
    ) -> typ.Generator[tuple[int, dict[str, typ.Any]], None, None]:  # type: ignore
        """Generate examples from the dataset."""
        for row in data.to_dicts():
            _hash = hash(frozenset(row.items()))
            yield _hash, row
