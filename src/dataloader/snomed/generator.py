"""Snomed: SNOMED CT Entity Linking Challenge."""

import json
import typing as typ

import datasets
import polars as pl

from dataloader.constants import PROJECT_ROOT

logger = datasets.logging.get_logger(__name__)

_CITATION = """
@article{hardman2023snomed,
    title = {SNOMED CT Entity Linking Challenge (version 1.0.0)},
    author = {Hardman, W. and Banks, M. and Davidson, R. and Truran, D. and Ayuningtyas, N. W. and Ngo, H. and Johnson,
    A. and Pollard, T.},
    year = {2023},
    journal = {PhysioNet},
    url = {https://doi.org/10.13026/s48e-sp45},
    doi = {10.13026/s48e-sp45}
}
"""

_DESCRIPTION = """
Snomed: A dataset for the SNOMED CT Entity Linking Challenge. The dataset consists of discharge notes
annotated with spans that correspond to SNOMED CT concept IDs.
Each annotation includes the start and end character offsets, SNOMED concept ID, and text content of the discharge note.
"""

_DATA_PATH = PROJECT_ROOT / "data/snomed/processed"


class SnomedConfig(datasets.BuilderConfig):
    """BuilderConfig for Snomed."""

    def __init__(self, **kwargs: typ.Any):
        """BuilderConfig for Snomed.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SnomedConfig, self).__init__(**kwargs)


class Snomed(datasets.GeneratorBasedBuilder):
    """Snomed: SNOMED CT Entity Linking Challenge Dataset."""

    BUILDER_CONFIGS = [
        SnomedConfig(
            name="snomed",
            version=datasets.Version("1.0.0", ""),
            description="SNOMED CT Entity Linking Challenge dataset.",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "note_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "annotations": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "concept_id": datasets.Value("int64"),
                            "concept_name": datasets.Value("string"),
                        }
                    ),
                    "classes": datasets.Value("string"),
                    "fewshots": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "concept_id": datasets.Value("int64"),
                            "concept_name": datasets.Value("string"),
                        }
                    ),
                }
            ),
            citation=_CITATION,
        )

    def aggregate_rows(self, data: pl.DataFrame) -> pl.DataFrame:
        """Process the MedDec data."""
        return data.group_by(["note_id", "text"]).agg(
            [
                # Collect all annotations as a list of dictionaries
                pl.struct(["start", "end", "concept_id", "concept_name"]).alias("annotations")
            ]
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:  # type: ignore
        data_path = dl_manager.download_and_extract(str(_DATA_PATH / "snomed.parquet"))

        data = pl.read_parquet(data_path)
        classes = {k: v for k, v in zip(data["concept_id"], data["concept_name"])}
        aggregated_data = self.aggregate_rows(data)
        aggregated_data = aggregated_data.with_columns([pl.lit(json.dumps(classes)).alias("classes")])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data": aggregated_data},
            ),
        ]

    def _generate_examples(  # type: ignore
        self, data: pl.DataFrame
    ) -> typ.Generator[tuple[int, dict[str, typ.Any]], None, None]:  # type: ignore
        """Generate examples from the dataset."""
        for row in data.to_dicts():
            yield row["note_id"], row
