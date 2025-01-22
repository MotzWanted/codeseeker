"""MIMIC-III-50: A public medical coding dataset from MIMIC-III with ICD-9 diagnosis and procedure codes."""

import hashlib
import json
import typing as typ

import datasets
import polars as pl

from dataloader import mimic_utils
from dataloader.constants import PROJECT_ROOT
from tools.code_trie import XMLTrie

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

_MEDICAL_CODING_SYSTEMS_DIR = PROJECT_ROOT / "data/medical-coding-systems/icd"
_MIMICIV_PATH = PROJECT_ROOT / "data/mimic-iv/processed/mimiciv.parquet"
_SPLITS_PATH = PROJECT_ROOT / "data/mimic-iv/splits/mimiciv_icd10_split.feather"


class MIMIC_IV_Config(datasets.BuilderConfig):
    """BuilderConfig for MIMIC-IV-ICD9."""

    def __init__(self, **kwargs: typ.Any):
        """BuilderConfig for MIMIC-IV-ICD9.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MIMIC_IV_Config, self).__init__(**kwargs)


class MIMIC_III_50(datasets.GeneratorBasedBuilder):
    """MIMIC-III-50: A public medical coding dataset from MIMIC-III with ICD-9 diagnosis
    and procedure codes Version 1.0."""

    BUILDER_CONFIGS = [
        MIMIC_IV_Config(
            name="icd9",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes and evidence spans.",
        ),
        MIMIC_IV_Config(
            name="icd10",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes and evidence spans.",
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

    def _split_generators(  # type: ignore
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:  # type: ignore
        mimic = self._process_data()
        classes = {
            str(code): desc
            for codes, descriptions in zip(mimic["codes"], mimic["code_descriptions"])
            for code, desc in zip(codes, descriptions)
        }
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

    def _get_code_descriptions(self, df: pl.DataFrame, code_column: str, code_lookup: dict[str, str]) -> pl.DataFrame:
        # Convert description DataFrame to dictionary for faster lookups
        return df.with_columns(
            pl.col(code_column)
            .map_elements(
                lambda codes: [code_lookup.get(code, "Unknown code") for code in codes],
                return_dtype=pl.List(pl.Utf8),
            )
            .alias(f"{code_column}_descriptions")
        )

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

        mimiciv_50 = mimic_utils.keep_top_k_codes(mimiciv, ["codes"], k=50).filter(pl.col("codes").is_not_null())

        mimiciv_50 = mimiciv_50.with_columns(
            pl.col("codes").map_elements(lambda x: list(set(x)), return_dtype=pl.List(pl.Utf8)).alias("codes")
        )

        if self.config.name == "icd9":
            code_descriptions = mimic_utils.get_icd9_descriptions(_MEDICAL_CODING_SYSTEMS_DIR)
            code_to_desc = dict(zip(code_descriptions["icd9_code"], code_descriptions["icd9_description"]))
            mimiciv_50 = self._get_code_descriptions(mimiciv_50, "codes", code_to_desc)
        elif self.config.name == "icd10":
            diag_xml_trie = XMLTrie.from_xml_file(
                _MEDICAL_CODING_SYSTEMS_DIR / "icd10cm_tabular_2025.xml", coding_system="icd10cm"
            )
            proc_xml_trie = XMLTrie.from_xml_file(
                _MEDICAL_CODING_SYSTEMS_DIR / "icd10pcs_tables_2025.xml", coding_system="icd10pcs"
            )

            mimiciv_50 = mimiciv_50.with_columns(
                pl.col("codes")
                .map_elements(
                    lambda codes: [
                        diag_xml_trie[code].desc if code in diag_xml_trie.lookup else proc_xml_trie[code].desc
                        for code in codes
                    ],
                    return_dtype=pl.List(pl.Utf8),
                )
                .alias("code_descriptions")
            )

        else:
            raise ValueError(f"Invalid code type: {self.config.name}")

        return mimiciv_50.filter(pl.col("codes").is_not_null())

    def _generate_examples(  # type: ignore
        self, data: pl.DataFrame
    ) -> typ.Generator[tuple[int, dict[str, typ.Any]], None, None]:  # type: ignore
        """Generate examples from a parquet file using split information."""
        data = data.drop("split")

        # Iterate through rows and yield examples
        for row in data.to_dicts():
            _hash = hashlib.md5(json.dumps(row).encode()).hexdigest()
            yield _hash, row
