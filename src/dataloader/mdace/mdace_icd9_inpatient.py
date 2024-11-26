"""MDACE-ICD9: MIMIC Documents Annotated with Code Evidence."""

from pathlib import Path
import typing as typ

import datasets
import polars as pl


logger = datasets.logging.get_logger(__name__)

_CITATION = """
@inproceedings{cheng-etal-2023-mdace,
    title = "{MDACE}: {MIMIC} Documents Annotated with Code Evidence",
    author = "Cheng, Hua  and
      Jafari, Rana  and
      Russell, April  and
      Klopfer, Russell  and
      Lu, Edmond  and
      Striner, Benjamin  and
      Gormley, Matthew",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.416",
    pages = "7534--7550",
}
"""

_DESCRIPTION = """
MDACE-ICD9: A medical coding dataset created from MDACE for inpatient records with ICD-9 diagnosis and procedure codes.
It includes annotated evidence spans, supports multi-label classification tasks,
and is split into train/validation/test sets.
"""

_PROJECT_ROOT = Path(__file__).parent.parent
_NOTES_PATH = _PROJECT_ROOT / "data/processed/mdace_notes.parquet"
_MIMICIV_PATH = _PROJECT_ROOT / "data/processed/mimiciv.parquet"
_ANNOTATIONS_PATH = _PROJECT_ROOT / "data/processed/mdace_inpatient_annotations.parquet"

_SPLITS = {
    "train": _PROJECT_ROOT / "data/splits/mdace/inpatient/MDace-ev-train.csv",
    "val": _PROJECT_ROOT / "data/splits/mdace/inpatient/MDace-ev-val.csv",
    "test": _PROJECT_ROOT / "data/splits/mdace/inpatient/MDace-ev-test.csv",
}


class MDACE_ICD9_Config(datasets.BuilderConfig):
    """BuilderConfig for MDACE-ICD9."""

    def __init__(self, **kwargs: typ.Any):
        """BuilderConfig for MDACE-ICD9.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MDACE_ICD9_Config, self).__init__(**kwargs)


class MDACE_ICD9(datasets.GeneratorBasedBuilder):
    """MDACE-ICD9: A dataset of inpatient records annotated with ICD-9 diagnosis and procedure codes."""

    BUILDER_CONFIGS = [
        MDACE_ICD9_Config(
            name="mdace-icd9-inpatient",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient dataset with ICD-9 codes and evidence spans.",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "note_id": datasets.Value("int64"),
                    "text": datasets.Value("string"),
                    "diagnosis_codes": datasets.Sequence(datasets.Value("string")),
                    "diagnosis_code_spans": datasets.Sequence(
                        datasets.Sequence(datasets.Sequence(datasets.Value("int64")))
                    ),
                    "diagnosis_code_type": datasets.Value("string"),
                    "procedure_codes": datasets.Sequence(datasets.Value("string")),
                    "procedure_code_spans": datasets.Sequence(
                        datasets.Sequence(datasets.Sequence(datasets.Value("int64")))
                    ),
                    "procedure_code_type": datasets.Value("string"),
                }
            ),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:  # type: ignore
        notes_path = dl_manager.download_and_extract(str(_NOTES_PATH))
        mimiciv_path = dl_manager.download_and_extract(str(_MIMICIV_PATH))
        annotations_path = dl_manager.download_and_extract(str(_ANNOTATIONS_PATH))

        splits = {split: dl_manager.download_and_extract(str(path)) for split, path in _SPLITS.items()}

        # Load and process the dataset
        data = self._process_data(notes_path, mimiciv_path, annotations_path)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data": data.filter(
                        pl.col("note_id").is_in(pl.read_csv(splits["train"], new_columns=["note_id"])["note_id"])
                    )
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data": data.filter(
                        pl.col("note_id").is_in(pl.read_csv(splits["val"], new_columns=["note_id"])["note_id"])
                    )
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data": data.filter(
                        pl.col("note_id").is_in(pl.read_csv(splits["test"], new_columns=["note_id"])["note_id"])
                    )
                },
            ),
        ]

    def _process_data(self, notes_path: str, mimiciv_path: str, annotations_path: str) -> pl.DataFrame:
        """Processes the raw dataset."""
        notes = pl.read_parquet(notes_path)
        mimiciv = pl.read_parquet(mimiciv_path)
        annotations = pl.read_parquet(annotations_path)

        # Filter annotations for ICD-9 codes
        annotations_icd9 = annotations.filter(pl.col("code_type").is_in({"icd9cm", "icd9pcs"}))

        # Generate mappings for ICD-9 code versions
        icd9cm_mapping = self._generate_code_mapping(
            mimiciv.filter(pl.col("diagnosis_code_type") == "icd9cm")["diagnosis_codes"].explode().unique(),
            annotations_icd9.filter(pl.col("code_type") == "icd9cm")["code"].explode().unique(),
        )
        icd9pcs_mapping = self._generate_code_mapping(
            mimiciv.filter(pl.col("procedure_code_type") == "icd9pcs")["procedure_codes"].explode().unique(),
            annotations_icd9.filter(pl.col("code_type") == "icd9pcs")["code"].explode().unique(),
        )

        # Map ICD-9 codes
        annotations_icd9 = annotations_icd9.with_columns(
            pl.col("code").map_elements(lambda code: icd9cm_mapping.get(code, code)).alias("code")
        )
        annotations_icd9 = annotations_icd9.with_columns(
            pl.col("code").map_elements(lambda code: icd9pcs_mapping.get(code, code)).alias("code")
        )

        # Aggregate annotations
        annotations_icd9_cm = (
            annotations_icd9.filter(pl.col("code_type") == "icd9cm")
            .group_by(["note_id"])
            .agg(
                [
                    pl.col("code").map_elements(list).alias("diagnosis_codes"),
                    pl.col("spans").map_elements(list).alias("diagnosis_code_spans"),
                    pl.col("code_type").last().alias("diagnosis_code_type"),
                ]
            )
        )
        annotations_icd9_pcs = (
            annotations_icd9.filter(pl.col("code_type") == "icd9pcs")
            .group_by(["note_id"])
            .agg(
                [
                    pl.col("code").map_elements(list).alias("procedure_codes"),
                    pl.col("spans").map_elements(list).alias("procedure_code_spans"),
                    pl.col("code_type").last().alias("procedure_code_type"),
                ]
            )
        )

        # Join annotations and notes
        annotations_icd9 = annotations_icd9_cm.join(annotations_icd9_pcs, on="note_id", how="outer_coalesce")
        data = notes.join(annotations_icd9, on="note_id", how="inner")

        # Final data adjustments
        data = data.with_columns(
            [
                pl.col("diagnosis_codes").fill_null([]),
                pl.col("procedure_codes").fill_null([]),
                pl.col("diagnosis_code_spans").fill_null([[]]),
                pl.col("procedure_code_spans").fill_null([[]]),
                pl.col("diagnosis_code_type").fill_null("icd9cm"),
                pl.col("procedure_code_type").fill_null("icd9pcs"),
            ]
        )

        return data

    def _generate_code_mapping(self, old_codes: set[str], new_codes: set[str]) -> dict[str, str]:
        """Generate a mapping from old ICD-9 codes to new ICD-9 codes."""
        mapping = {}
        for new_code in new_codes:
            temp_code = new_code
            while temp_code and temp_code not in old_codes:
                temp_code = temp_code[:-1]
            mapping[new_code] = temp_code if temp_code else new_code
        return mapping

    def _generate_examples(  # type: ignore
        self, data: pl.DataFrame
    ) -> typ.Generator[tuple[int, dict[str, typ.Any]], None, None]:  # type: ignore
        """Generate examples from the dataset."""
        for row in data.to_dicts():
            yield row["note_id"], row
