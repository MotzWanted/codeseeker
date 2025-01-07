"""MDACE-ICD: MIMIC Documents Annotated with Code Evidence."""

import json
import typing as typ

import datasets
import polars as pl

from dataloader import mimic_utils
from dataloader.constants import PROJECT_ROOT
from tools.code_trie import XMLTrie


logger = datasets.logging.get_logger(__name__)

_ANNOTATIONS_PATH = PROJECT_ROOT / "data/mdace/processed/mdace_inpatient_annotations.parquet"
_MEDICAL_CODING_SYSTEMS_DIR = PROJECT_ROOT / "data/medical-coding-systems/icd"

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
            name="icd10cm-3.0",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes of 3 digits and evidence spans.",
        ),
        MDACEICD9Config(
            name="icd10cm-3.1",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes of 4 digits and evidence spans.",
        ),
        MDACEICD9Config(
            name="icd10cm-3.2",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes of 5 digits and evidence spans.",
        ),
        MDACEICD9Config(
            name="icd10cm-3.3",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes of 6 digits and evidence spans.",
        ),
        MDACEICD9Config(
            name="icd10cm-3.4",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes of 7 digits and evidence spans.",
        ),
        MDACEICD9Config(
            name="icd10pcs-4.0",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 procedure codes of 4 digits and evidence spans.",
        ),
        MDACEICD9Config(
            name="icd10pcs-4.1",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 procedure codes of 5 digits and evidence spans.",
        ),
        MDACEICD9Config(
            name="icd10pcs-4.2",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 procedure codes of 6 digits and evidence spans.",
        ),
        MDACEICD9Config(
            name="icd10pcs-4.3",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 procedure codes of 7 digits and evidence spans.",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    mimic_utils.ID_COLUMN: datasets.Value("int64"),
                    mimic_utils.ROW_ID_COLUMN: datasets.Value("string"),
                    "note_type": datasets.Value("string"),
                    "note_subtype": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "annotations": datasets.Sequence(
                        {
                            "code": datasets.Value("string"),
                            "code_type": datasets.Value("string"),
                            "code_description": datasets.Value("string"),
                            "spans": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                        }
                    ),
                    "classes": datasets.Value("string"),
                }
            ),
            citation=_CITATION,
        )

    def aggregate_rows(self, data: pl.DataFrame) -> pl.DataFrame:
        """Process the MedDec data."""
        return data.group_by(
            [
                mimic_utils.ID_COLUMN,
                mimic_utils.ROW_ID_COLUMN,
                "note_type",
                "note_subtype",
                "text",
            ]
        ).agg(
            [
                # Collect all annotations as a list of dictionaries
                pl.struct(
                    [
                        "code",
                        "code_type",
                        "code_description",
                        "spans",
                    ]
                ).alias("annotations")
            ]
        )

    def parse_config_name(self) -> tuple[str, int]:
        """Parse the configuration name."""
        code_type, code_level = self.config.name.split("-")
        code_level = int(code_level.split(".")[-1])
        return code_type, code_level

    def _split_generators(  # type: ignore
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:
        # splits = {split: dl_manager.download_and_extract(str(path)) for split, path in _SPLITS.items()}
        splits = {split: pl.read_csv(path, new_columns=[mimic_utils.ID_COLUMN]) for split, path in _SPLITS.items()}

        code_type, code_level = self.parse_config_name()
        data = self._process_data(
            code_level=int(code_level),
            code_type=code_type,
        )
        classes = {k: v for k, v in zip(data["code"], data["code_description"])}
        aggregated_data = self.aggregate_rows(data)
        aggregated_data = aggregated_data.with_columns([pl.lit(json.dumps(classes)).alias("classes")])
        # Ensure note_id is the same type in both `splits` and `data`
        splits = {
            k: v.with_columns(pl.col(mimic_utils.ID_COLUMN).cast(pl.Int64)) for k, v in splits.items()
        }  # Cast to Int64

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data": aggregated_data.filter(
                        pl.col(mimic_utils.ID_COLUMN).is_in(splits["train"][mimic_utils.ID_COLUMN])
                    )
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data": aggregated_data.filter(
                        pl.col(mimic_utils.ID_COLUMN).is_in(splits["val"][mimic_utils.ID_COLUMN])
                    )
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data": aggregated_data.filter(
                        pl.col(mimic_utils.ID_COLUMN).is_in(splits["test"][mimic_utils.ID_COLUMN])
                    )
                },
            ),
        ]

    def _process_data(
        self,
        code_type: str,
        code_level: int,
    ) -> pl.DataFrame:
        """Processes the raw dataset."""
        mdace = pl.read_parquet(_ANNOTATIONS_PATH)

        if "cm" in code_type:
            code_system_file = f"{code_type}_tabular_2025.xml"
        elif "pcs" in code_type:
            code_system_file = f"{code_type}_tables_2025.xml"
        else:
            raise ValueError(f"Invalid code type: {code_type}")

        xml_trie = XMLTrie.from_xml_file(_MEDICAL_CODING_SYSTEMS_DIR / code_system_file, coding_system=code_type)

        # Filter annotations for ICD codes
        mdace_icd = mdace.filter(pl.col("code_type").is_in({code_type}))

        # Cast note_id to Int64 to match the type of _id in mimiciii
        mdace_icd = mdace_icd.with_columns(pl.col(mimic_utils.ROW_ID_COLUMN).cast(pl.Int64))

        # Transform spans to list[list[int]] format
        mdace_icd = mdace_icd.with_columns(
            pl.col("spans")
            .map_elements(
                lambda span_list: [[span["start"], span["end"]] for span in span_list],
                return_dtype=pl.List(pl.List(pl.Int64)),
            )
            .alias("spans")
        )

        def find_description(code: str, code_level: int) -> tuple[str, str]:
            """Find the description of a code."""
            if code in xml_trie.lookup:
                return code, xml_trie[code].desc
            # If the code is still valid for truncation, recurse with one less level
            if len(code) < 3:
                # If all levels are exhausted, raise an error or return a fallback value
                raise ValueError(f"Code {code} not found in the XML trie.")
            code_level -= 1
            truncated_code = mimic_utils.truncate_code(code, code_level)
            return find_description(truncated_code, code_level)

        truncated_annotations_icd = mdace_icd.with_columns(
            pl.col("code")
            .map_elements(
                lambda code: mimic_utils.truncate_code(code, code_level),
                return_dtype=pl.Utf8,
            )
            .alias("code")
        )

        truncated_annotations_icd = truncated_annotations_icd.with_columns(
            [
                pl.col("code")
                .map_elements(lambda x: find_description(x, code_level)[0], return_dtype=pl.Utf8)
                .alias("code"),
                pl.col("code")
                .map_elements(lambda x: find_description(x, code_level)[1], return_dtype=pl.Utf8)
                .alias("code_description"),
            ]
        )

        return truncated_annotations_icd

    def _generate_code_mapping(self, old_codes: set[str], new_codes: set[str]) -> dict[str, str]:
        """Generate a mapping from old ICD-9 codes to new ICD-10 codes."""
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
