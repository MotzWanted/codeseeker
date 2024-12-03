"""MDACE-ICD9: MIMIC Documents Annotated with Code Evidence."""

import typing as typ

import datasets
import polars as pl

from dataloader import mimic_utils
from dataloader.constants import PROJECT_ROOT


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

_NOTES_PATH = PROJECT_ROOT / "data/mdace/processed/mdace_notes.parquet"
_MIMICIII_PATH = PROJECT_ROOT / "data/mimic-iii/processed/mimiciii.parquet"
_ANNOTATIONS_PATH = PROJECT_ROOT / "data/mdace/processed/mdace_inpatient_annotations.parquet"

_SPLITS = {
    "train": PROJECT_ROOT / "data/mdace/splits/MDace-ev-train.csv",
    "val": PROJECT_ROOT / "data/mdace/splits/MDace-ev-val.csv",
    "test": PROJECT_ROOT / "data/mdace/splits/MDace-ev-test.csv",
}


class MDACEICD9Config(datasets.BuilderConfig):
    """BuilderConfig for MDACE-ICD9."""

    def __init__(self, **kwargs: typ.Any):
        """BuilderConfig for MDACE-ICD9.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MDACEICD9Config, self).__init__(**kwargs)


class MDACE_ICD9(datasets.GeneratorBasedBuilder):
    """MDACE-ICD9: A dataset of inpatient records annotated with ICD-9 diagnosis and procedure codes."""

    BUILDER_CONFIGS = [
        MDACEICD9Config(
            name="icd9-diagnosis",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-9 diagnosis codes and evidence spans.",
        ),
        MDACEICD9Config(
            name="icd9-procedure",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-9 procedure codes and evidence spans.",
        ),
        MDACEICD9Config(
            name="icd10-diagnosis",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes and evidence spans.",
        ),
        MDACEICD9Config(
            name="icd10-procedure",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 procedure codes and evidence spans.",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "subject_id": datasets.Value("int64"),
                    "_id": datasets.Value("int64"),
                    "note_id": datasets.Value("string"),
                    "note_type": datasets.Value("string"),
                    "note_subtype": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "codes": datasets.Sequence(datasets.Value("string")),
                    "type": datasets.Value("string"),
                    "spans": datasets.Sequence(
                        datasets.Sequence(datasets.Sequence(datasets.Value("int64")))
                    ),  # Adjust if the nesting is different
                }
            ),
            citation=_CITATION,
        )

    def _split_generators(  # type: ignore
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:
        # splits = {split: dl_manager.download_and_extract(str(path)) for split, path in _SPLITS.items()}
        splits = {split: pl.read_csv(path, new_columns=[mimic_utils.ID_COLUMN]) for split, path in _SPLITS.items()}

        icd_type, code_type = self.config.name.split("-")
        data = self._process_data(
            notes_path=_NOTES_PATH,
            mimiciii_path=_MIMICIII_PATH,
            annotations_path=_ANNOTATIONS_PATH,
            code_type=code_type,
            icd_type=icd_type,
        )

        # Ensure note_id is the same type in both `splits` and `data`
        splits = {
            k: v.with_columns(pl.col(mimic_utils.ID_COLUMN).cast(pl.Int64)) for k, v in splits.items()
        }  # Cast to Int64

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data": data.filter(pl.col(mimic_utils.ID_COLUMN).is_in(splits["train"][mimic_utils.ID_COLUMN]))
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data": data.filter(pl.col(mimic_utils.ID_COLUMN).is_in(splits["val"][mimic_utils.ID_COLUMN]))
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data": data.filter(pl.col(mimic_utils.ID_COLUMN).is_in(splits["test"][mimic_utils.ID_COLUMN]))
                },
            ),
        ]

    def _process_data(
        self, notes_path: str, mimiciii_path: str, annotations_path: str, icd_type: str, code_type: str
    ) -> pl.DataFrame:
        """Processes the raw dataset."""
        notes = pl.read_parquet(notes_path)
        mimiciii = pl.read_parquet(mimiciii_path)
        mdace = pl.read_parquet(annotations_path)

        if code_type == "diagnosis":
            code_key = f"{icd_type}cm"
        elif code_type == "procedure":
            code_key = f"{icd_type}pcs"
        else:
            raise ValueError(f"Invalid code type: {code_type}")

        # Filter annotations for ICD-9 codes
        mdace_icd9 = mdace.filter(pl.col("code_type").is_in({code_key}))

        # Cast note_id to Int64 to match the type of _id in mimiciii
        mdace_icd9 = mdace_icd9.with_columns(pl.col(mimic_utils.ROW_ID_COLUMN).cast(pl.Int64))

        # Transform spans to list[list[int]] format
        mdace_icd9 = mdace_icd9.with_columns(
            pl.col("spans")
            .map_elements(
                lambda span_list: [[span["start"], span["end"]] for span in span_list],
                return_dtype=pl.List(pl.List(pl.Int64)),
            )
            .alias("spans")
        )

        # Generate mappings for ICD-9 code versions
        icd_mapping = self._generate_code_mapping(
            mimiciii.filter(pl.col(f"{code_type}_code_type") == code_key)[f"{code_type}_codes"].explode().unique(),
            mdace_icd9.filter(pl.col("code_type") == code_key)["code"].explode().unique(),
        )

        # Map ICD codes with explicit return_dtype
        annotations_icd9 = mdace_icd9.with_columns(
            pl.col("code").map_elements(lambda code: icd_mapping.get(code, code), return_dtype=pl.Utf8).alias("code")
        )

        # Aggregate annotations
        annotations_icd = (
            annotations_icd9.filter(pl.col("code_type") == code_key)
            .group_by([mimic_utils.ROW_ID_COLUMN])
            .agg(
                pl.col("code").map_elements(list, return_dtype=pl.List(pl.Utf8)),
                pl.col("spans").map_elements(list),
                pl.col("code_type").last(),
            )
        )

        notes = notes.with_columns(pl.col(mimic_utils.ROW_ID_COLUMN).cast(pl.Int64))

        # Join annotations and notes
        return notes.join(annotations_icd, on=mimic_utils.ROW_ID_COLUMN, how="inner")

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
            yield row[mimic_utils.ROW_ID_COLUMN], row
