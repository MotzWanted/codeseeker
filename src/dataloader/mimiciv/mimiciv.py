"""MIMIC-IV-ICD10: A medical coding dataset extracted from MIMIC-IV with ICD-10 diagnosis and procedure codes."""

import hashlib
import json
import typing as typ

import datasets
import polars as pl

from dataloader import mimic_utils
from dataloader.constants import PROJECT_ROOT
from tools.code_trie import XMLTrie, add_hard_negatives_to_set

logger = datasets.logging.get_logger(__name__)

_MEDICAL_CODING_SYSTEMS_DIR = PROJECT_ROOT / "data/medical-coding-systems/icd"
_MIMICIV_PATH = PROJECT_ROOT / "data/mimic-iv/processed/mimiciv.parquet"
_SPLITS_PATH = PROJECT_ROOT / "data/mimic-iv/splits/mimiciv_icd10_split.feather"

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
MIMIC-IV: A medical coding dataset created from MIMIC-IV with ICD-10 and ICD-9 diagnosis and procedure codes.
This dataset is processed to retain only relevant ICD-10 codes, filter rare codes, and ensure no duplicate entries.
It also includes train/validation/test splits.
"""


class MIMIC_IV_Config(datasets.BuilderConfig):
    """BuilderConfig for MIMIC-IV-ICD9."""

    def __init__(self, **kwargs: typ.Any):
        """BuilderConfig for MIMIC-IV-ICD9.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MIMIC_IV_Config, self).__init__(**kwargs)


class MIMIC_IV_ICD10(datasets.GeneratorBasedBuilder):
    """MIMIC-IV-ICD10: A medical coding dataset with ICD-10 diagnosis and procedure codes."""

    BUILDER_CONFIGS = [
        MIMIC_IV_Config(
            name="icd10pcs-4.0",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes and evidence spans.",
        ),
        MIMIC_IV_Config(
            name="icd10pcs-4.1",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes and evidence spans.",
        ),
        MIMIC_IV_Config(
            name="icd10pcs-4.2",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes and evidence spans.",
        ),
        MIMIC_IV_Config(
            name="icd10pcs-4.3",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes and evidence spans.",
        ),
        MIMIC_IV_Config(
            name="icd10cm-3.0",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes of 3 digits and evidence spans.",
        ),
        MIMIC_IV_Config(
            name="icd10cm-3.1",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes of 4 digits and evidence spans.",
        ),
        MIMIC_IV_Config(
            name="icd10cm-3.2",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes of 5 digits and evidence spans.",
        ),
        MIMIC_IV_Config(
            name="icd10cm-3.3",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes of 6 digits and evidence spans.",
        ),
        MIMIC_IV_Config(
            name="icd10cm-3.4",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes of 7 digits and evidence spans.",
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

    def parse_config_name(self) -> tuple[str, int]:
        """Parse the configuration name."""
        code_type, code_level = self.config.name.split("-")
        code_level = int(code_level.split(".")[-1])
        return code_type, code_level

    def _split_generators(  # type: ignore
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:  # type: ignore
        code_type, code_level = self.parse_config_name()
        mimic = self._process_data(code_level=code_level, code_type=code_type)
        classes = {
            code: desc
            for codes, descriptions in zip(mimic["codes"], mimic["code_descriptions"])
            for code, desc in zip(codes, descriptions)
        }
        classes.update(
            {
                code: desc
                for codes, descriptions in zip(mimic["negatives"], mimic["negative_descriptions"])
                for code, desc in zip(codes, descriptions)
            }
        )
        mimic = mimic.with_columns([pl.lit(json.dumps(classes)).alias("classes")])
        # Load split information
        splits = pl.read_ipc(_SPLITS_PATH)
        splits = splits.rename(
            {
                "_id": mimic_utils.ID_COLUMN,
            }
        )
        data = mimic.join(splits, on=mimic_utils.ID_COLUMN)

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

    def _process_data(
        self,
        code_type: str,
        code_level: int,
    ) -> pl.DataFrame:
        """Processes the raw dataset."""
        mimiciv = pl.read_parquet(_MIMICIV_PATH)

        if "cm" in code_type:
            mimic_code_type_key = "diagnosis"
            code_system_file = f"{code_type}_tabular_2025.xml"
        elif "pcs" in code_type:
            mimic_code_type_key = "procedure"
            code_system_file = f"{code_type}_tables_2025.xml"
        else:
            raise ValueError(f"Invalid code type: {code_type}")

        xml_trie = XMLTrie.from_xml_file(_MEDICAL_CODING_SYSTEMS_DIR / code_system_file, coding_system=code_type)

        mimiciv = mimiciv.with_columns(
            pl.col(f"{mimic_code_type_key}_code_type").alias("code_type"),
            pl.col(f"{mimic_code_type_key}_codes").alias("codes"),
        )

        mimiciv_filtered = mimiciv.filter(pl.col("code_type").str.contains(code_type))

        mimiciv_clean = mimic_utils.remove_rare_codes(mimiciv_filtered, code_columns=["codes"], min_count=10)

        # Truncate codes to `code_level` and look up descriptions
        truncated_mimiciv = mimiciv_clean.with_columns(
            pl.col("codes")
            .map_elements(
                lambda codes: list(set([mimic_utils.truncate_code(code, code_level) for code in codes])),
                return_dtype=pl.List(pl.Utf8),
            )
            .alias("codes")
        )
        truncated_mimiciv = truncated_mimiciv.with_columns(
            pl.col("codes")
            .map_elements(
                lambda codes: [
                    xml_trie[code].desc if code in xml_trie.lookup else xml_trie[code[:3]].desc for code in codes
                ],
                return_dtype=pl.List(pl.Utf8),
            )
            .alias("code_descriptions")
        )

        truncated_mimiciv_with_negatives = add_hard_negatives_to_set(truncated_mimiciv, xml_trie, "codes", "negatives")
        # Truncate negatives to `code_level` and look up descriptions
        truncated_mimiciv_with_negatives = truncated_mimiciv_with_negatives.with_columns(
            pl.col("negatives")
            .map_elements(
                lambda codes: list(set([mimic_utils.truncate_code(code, code_level) for code in codes])),
                return_dtype=pl.List(pl.Utf8),
            )
            .alias("negatives")
        )

        truncated_mimiciv_with_negatives = truncated_mimiciv_with_negatives.with_columns(
            pl.col("negatives")
            .map_elements(
                lambda codes: [
                    xml_trie[code].desc if code in xml_trie.lookup else xml_trie[code[:3]].desc for code in codes
                ],
                return_dtype=pl.List(pl.Utf8),
            )
            .alias("negative_descriptions")
        )

        return truncated_mimiciv_with_negatives.filter(pl.col("codes").is_not_null())

    def _generate_examples(  # type: ignore
        self, data: pl.DataFrame
    ) -> typ.Generator[tuple[int, dict[str, typ.Any]], None, None]:  # type: ignore
        """Generate examples from a parquet file using split information."""
        data = data.drop("split")

        # Iterate through rows and yield examples
        for row in data.to_dicts():
            _hash = hashlib.md5(json.dumps(row).encode()).hexdigest()
            yield _hash, row
