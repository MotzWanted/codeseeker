"""MIMIC-IV-ICD10: A medical coding dataset extracted from MIMIC-IV with ICD-10 diagnosis and procedure codes."""

import hashlib
import json
import typing as typ

import datasets
import polars as pl

from dataloader import mimic_utils
from dataloader.constants import PROJECT_ROOT
from tools.code_trie import XMLTrie

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
            name="icd9",
            version=datasets.Version("1.0.0", ""),
            description="MIMIC IV patient dataset with ICD-9 diagnosis and procedure codes.",
        ),
        MIMIC_IV_Config(
            name="icd10",
            version=datasets.Version("1.0.0", ""),
            description="MIMIC IV patient dataset with ICD-10 diagnosis and procedure codes.",
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
                }
            ),
            citation=_CITATION,
        )

    def _split_generators(  # type: ignore
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:  # type: ignore
        mimic = self._process_data()
        # Load split information
        splits = pl.read_ipc(_SPLITS_PATH, memory_map=False)
        splits = splits.rename(
            {
                "_id": mimic_utils.ID_COLUMN,
            }
        )
        data = mimic.join(splits, on=mimic_utils.ID_COLUMN)
        # data = data.with_columns([pl.lit(json.dumps(classes)).alias("classes")])
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

    def _process_data(self) -> pl.DataFrame:
        """Processes the raw dataset."""
        mimiciv = pl.read_parquet(_MIMICIV_PATH)

        # Filter for code_type
        mimiciv = mimiciv.filter(
            pl.col("diagnosis_code_type").str.contains(self.config.name)
            | pl.col("procedure_code_type").str.contains(self.config.name)
        )

        mimiciv = mimiciv.with_columns(
            pl.concat_list([pl.col("diagnosis_codes").fill_null([]), pl.col("procedure_codes").fill_null([])]).alias(
                "codes"
            )
        )

        mimiciv_clean = mimic_utils.remove_rare_codes(mimiciv, code_columns=["codes"], min_count=100)

        if self.config.name == "icd9":
            code_descriptions = mimic_utils.get_icd9_descriptions(_MEDICAL_CODING_SYSTEMS_DIR)
            code_to_desc = dict(zip(code_descriptions["icd9_code"], code_descriptions["icd9_description"]))
            mimiciv = self._get_code_descriptions(mimiciv_clean, "codes", code_to_desc)
        elif self.config.name == "icd10":
            diag_xml_trie = XMLTrie.from_xml_file(
                _MEDICAL_CODING_SYSTEMS_DIR / "icd10cm_tabular_2025.xml", coding_system="icd10cm"
            )
            proc_xml_trie = XMLTrie.from_xml_file(
                _MEDICAL_CODING_SYSTEMS_DIR / "icd10pcs_tables_2025.xml", coding_system="icd10pcs"
            )

            mimiciv = mimiciv.with_columns(
                pl.col("codes")
                .map_elements(
                    lambda codes: [
                        diag_xml_trie[code].desc
                        if code in diag_xml_trie.lookup
                        else proc_xml_trie[code].desc
                        if code in proc_xml_trie.lookup
                        else None
                        for code in codes
                    ],
                    return_dtype=pl.List(pl.Utf8),
                )
                .alias("code_descriptions")
            )
            # loop over codes for each row and filter out those codes whose descriptions are None based on the list
            mimiciv = mimiciv.with_columns(
                pl.col("codes").map_elements(
                    lambda codes: [
                        code for code in codes if code in diag_xml_trie.lookup or code in proc_xml_trie.lookup
                    ],
                    return_dtype=pl.List(pl.Utf8),
                )
            )

        else:
            raise ValueError(f"Invalid code type: {self.config.name}")

        # filter out those codes (pl.col("codes")) whose descriptions (pl.col("code_descriptions")) are None

        return mimiciv.drop(
            [
                "diagnosis_codes",
                "procedure_codes",
                "diagnosis_code_type",
                "procedure_code_type",
                "charttime",
                "storetime",
                "note_type",
                "note_seq",
            ]
        )

    def _generate_examples(  # type: ignore
        self, data: pl.DataFrame
    ) -> typ.Generator[tuple[int, dict[str, typ.Any]], None, None]:  # type: ignore
        """Generate examples from a parquet file using split information."""
        data = data.drop("split")

        # Iterate through rows and yield examples
        for row in data.to_dicts():
            _hash = hashlib.md5(json.dumps(row).encode()).hexdigest()
            yield _hash, row
