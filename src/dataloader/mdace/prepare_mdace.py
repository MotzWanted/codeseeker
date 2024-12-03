"""
NOTE: This script is modified from https://github.com/JoakimEdin/explainable-medical-coding/blob/main/explainable_medical_coding/data/prepare_mdace.py

This script takes the raw data from the MIMIC-III dataset and prepares it for automatic medical coding.
The script does the following:
1. Loads the data from the csv files.
2. Renames the columns to match the column names in the MIMIC-IV dataset.
3. Adds punctuations to the ICD-9-CM and ICD-9-PCS codes.
4. Joins discharge summaries with addendums.
5. Removes duplicate rows.
6. Removes cases with no codes.
7. Saves the data as parquet files.

MIMIC-III have many different types of notes that are ignored in this script.
The notes are stored in the NOTEEVENTS.csv file. Here are the note categories and their counts:
┌───────────────────┬────────┐
│ CATEGORY          ┆ counts │
│ ---               ┆ ---    │
│ str               ┆ u32    │
╞═══════════════════╪════════╡
│ Discharge summary ┆ 59652  │
│ Physician         ┆ 141624 │
│ Case Management   ┆ 967    │
│ Consult           ┆ 98     │
│ Nursing           ┆ 223556 │
│ General           ┆ 8301   │
│ Respiratory       ┆ 31739  │
│ Echo              ┆ 45794  │
│ Social Work       ┆ 2670   │
│ Radiology         ┆ 522279 │
│ Nursing/other     ┆ 822497 │
│ Nutrition         ┆ 9418   │
│ Pharmacy          ┆ 103    │
│ ECG               ┆ 209051 │
│ Rehab Services    ┆ 5431   │
└───────────────────┴────────┘
These notes may be useful for other tasks. For example, for pre-training language models.
It is also not guaranteed that all the information is in the discharge summaries.
"""

import json
import logging
import random
import string
from collections import defaultdict
from pathlib import Path

import polars as pl

from dataloader import mimic_utils

random.seed(10)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path("data/mdace/processed")


def parse_code_dataframe(
    df: pl.DataFrame,
    code_column: str = "diagnosis_codes",
    code_type_column: str = "diagnosis_code_type",
) -> pl.DataFrame:
    """Change names of colums, remove duplicates and Nans, and takes a dataframe and a column name
    and returns a series with the column name and a list of codes.

    Example:
        Input:
                subject_id  _id     target
                       2   163353     V3001
                       2   163353      V053
                       2   163353      V290

        Output:
            target    [V053, V290, V3001]

    Args:
        row (pd.DataFrame): Dataframe with a column of codes.
        col (str): column name of the codes.

    Returns:
        pd.Series: Series with the column name and a list of codes.
    """

    df = df.filter(df[code_column].is_not_null())
    df = df.unique(subset=[mimic_utils.ID_COLUMN, code_column])
    df = df.group_by([mimic_utils.ID_COLUMN, code_type_column]).agg(
        pl.col(code_column).map_elements(list, return_dtype=pl.List(pl.Utf8)).alias(code_column)
    )
    return df


def get_mdace_annotations(path: Path) -> pl.DataFrame:
    rows = []
    for json_path in path.glob("**/*.json"):
        with open(json_path, "r", encoding="utf8") as json_file:
            case_annotations = json.load(json_file)
            hadm_id = case_annotations["hadm_id"]

            for note in case_annotations["notes"]:
                note_id = note["note_id"]
                note_category = note["category"]
                note_description = note["description"]

                code2spans = defaultdict(list)  # code -> list of spans
                code2system = {}  # code -> code system (e.g. ICD-9, ICD-10, etc.)

                for annotation in note["annotations"]:
                    code = annotation["code"]
                    code2system[code] = annotation["code_system"]
                    code2spans[code].append((annotation["begin"], annotation["end"]))

                for code, spans in code2spans.items():
                    rows.append(
                        (
                            hadm_id,
                            note_id,
                            note_category,
                            note_description,
                            code2system[code],
                            code,
                            spans,
                        )
                    )
                # print(code_dict)

    schema = {
        mimic_utils.ID_COLUMN: pl.Int64,
        "note_id": pl.Int64,
        "note_type": pl.Utf8,
        "note_subtype": pl.Utf8,
        "code_type": pl.Utf8,
        "code": pl.Utf8,
        "spans": pl.List,
    }
    return pl.DataFrame(schema=schema, data=rows, orient="row")


def trim_annotations(
    span: tuple[int, int],
    text: str,
    punctuations: set[str] = set(string.punctuation + "\n\t "),
) -> tuple[int, int]:
    start = span[0]
    end = span[1]

    if text[end] in punctuations:
        end -= 1

    if text[start] in punctuations:
        start += 1

    return start, end


def clean_mdace_annotations(mdace_annotations: pl.DataFrame, mdace_notes: pl.DataFrame) -> pl.DataFrame:
    mdace_annotations = mdace_annotations.join(mdace_notes[["note_id", "text"]], on="note_id", how="inner")
    mdace_annotations = mdace_annotations.with_columns(
        spans=pl.struct("text", "spans").map_elements(
            lambda row: [trim_annotations(span, row["text"]) for span in row["spans"]],
            return_dtype=pl.List(pl.Struct([pl.Field("start", pl.Int64), pl.Field("end", pl.Int64)])),
        )
    )

    return mdace_annotations


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info("Preparing the MDace dataset from raw data")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the dataframes
    mimic_notes = pl.read_csv(PROJECT_ROOT / "data/mimic-iii/raw/NOTEEVENTS.csv.gz")
    mimic_notes = mimic_notes.rename(
        {
            "HADM_ID": mimic_utils.ID_COLUMN,
            "SUBJECT_ID": mimic_utils.SUBJECT_ID_COLUMN,
            "ROW_ID": mimic_utils.ROW_ID_COLUMN,
            "TEXT": mimic_utils.TEXT_COLUMN,
            "CATEGORY": "note_type",
            "DESCRIPTION": "note_subtype",
        }
    )

    mdace_inpatient_annotations = get_mdace_annotations(Path("data/mdace/raw/Inpatient"))
    mdace_profee_annotations = get_mdace_annotations(Path("data/mdace/raw/Profee"))
    mdace_notes = mimic_notes.filter(pl.col("note_id").is_in(mdace_inpatient_annotations["note_id"]))
    mdace_inpatient_annotations = clean_mdace_annotations(mdace_inpatient_annotations, mdace_notes)
    mdace_profee_annotations = clean_mdace_annotations(mdace_profee_annotations, mdace_notes)

    mdace_inpatient_annotations = mdace_inpatient_annotations.with_columns(
        pl.col("code_type")
        .str.replace("ICD-9-CM", "icd9cm")
        .str.replace("ICD-10-CM", "icd10cm")
        .str.replace("ICD-10-PCS", "icd10pcs")
        .str.replace("CPT", "cpt")
        .str.replace("ICD-9-PCS", "icd9pcs")
    )

    mdace_profee_annotations = mdace_profee_annotations.with_columns(
        pl.col("code_type")
        .str.replace("ICD-9-CM", "icd9cm")
        .str.replace("ICD-10-CM", "icd10cm")
        .str.replace("ICD-10-PCS", "icd10pcs")
        .str.replace("CPT", "cpt")
        .str.replace("ICD-9-PCS", "icd9pcs")
    )

    # convert note_id to string
    mdace_notes = mdace_notes.with_columns(
        note_id=pl.col("note_id").cast(pl.Utf8),
    )
    mdace_inpatient_annotations = mdace_inpatient_annotations.with_columns(
        note_id=pl.col("note_id").cast(pl.Utf8),
    )
    mdace_profee_annotations = mdace_profee_annotations.with_columns(
        note_id=pl.col("note_id").cast(pl.Utf8),
    )

    # save files to disk
    logger.info(f"Saving the MDace dataset to {OUTPUT_DIR}")
    mdace_notes.write_parquet(OUTPUT_DIR / "mdace_notes.parquet")
    mdace_inpatient_annotations.write_parquet(OUTPUT_DIR / "mdace_inpatient_annotations.parquet")
    mdace_profee_annotations.write_parquet(OUTPUT_DIR / "mdace_profee_annotations.parquet")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
