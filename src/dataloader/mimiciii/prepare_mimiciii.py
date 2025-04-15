"""
NOTE: This script is modified from https://github.com/JoakimEdin/explainable-medical-coding/blob/main/explainable_medical_coding/data/prepare_mimiciii.py

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

import logging
import random
from pathlib import Path

import polars as pl

from dataloader import mimic_utils

random.seed(10)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path("data/mimic-iii/processed")


TOP_50_MULLENBACH_CODES = {
    "401.9",
    "38.93",
    "428.0",
    "427.31",
    "414.01",
    "96.04",
    "96.6",
    "584.9",
    "250.00",
    "96.71",
    "272.4",
    "518.81",
    "99.04",
    "39.61",
    "599.0",
    "530.81",
    "96.72",
    "272.0",
    "285.9",
    "88.56",
    "244.9",
    "486",
    "38.91",
    "285.1",
    "36.15",
    "276.2",
    "496",
    "99.15",
    "995.92",
    "V58.61",
    "507.0",
    "038.9",
    "88.72",
    "585.9",
    "403.90",
    "311",
    "305.1",
    "37.22",
    "412",
    "33.24",
    "39.95",
    "287.5",
    "410.71",
    "276.1",
    "V45.81",
    "424.0",
    "45.13",
    "V15.82",
    "511.9",
    "37.23",
}


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


def parse_notes_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Parse the notes dataframe by filtering out notes with no text and removing duplicates."""
    df = df.filter(df["note_type"] == "Discharge summary")
    df = df.filter(df[mimic_utils.TEXT_COLUMN].is_not_null())
    df = df.sort(
        [mimic_utils.ID_COLUMN, "note_subtype", "CHARTTIME", "CHARTDATE", "note_id"],
        descending=[False, True, False, False, False],
    )
    # join the notes with the same id and note type. This is to join discharge summaries and addendums.
    df = df.group_by(mimic_utils.SUBJECT_ID_COLUMN, mimic_utils.ID_COLUMN, "note_type").agg(
        pl.col("text").str.concat(" "),
        pl.col("note_subtype").str.to_lowercase().str.concat("+"),
        pl.col("note_id").str.concat("+"),
    )
    return df


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info("Preparing the MIMIC-III dataset from raw data")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the dataframes
    mimic_notes = pl.read_csv(PROJECT_ROOT / "data/mimic-iii/raw/NOTEEVENTS.csv.gz")
    mimic_diag = pl.read_csv(PROJECT_ROOT / "data/mimic-iii/raw/DIAGNOSES_ICD.csv.gz", dtypes={"ICD9_CODE": str})
    mimic_proc = pl.read_csv(PROJECT_ROOT / "data/mimic-iii/raw/PROCEDURES_ICD.csv.gz", dtypes={"ICD9_CODE": str})

    # rename the columns
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
    mimic_diag = mimic_diag.rename(
        {
            "HADM_ID": mimic_utils.ID_COLUMN,
            "ICD9_CODE": "diagnosis_codes",
        }
    )
    mimic_proc = mimic_proc.rename(
        {
            "HADM_ID": mimic_utils.ID_COLUMN,
            "ICD9_CODE": "procedure_codes",
        }
    )

    # Format the code type columns
    mimic_diag = mimic_diag.with_columns(diagnosis_code_type=pl.lit("icd9cm"))

    mimic_proc = mimic_proc.with_columns(procedure_code_type=pl.lit("icd9pcs"))

    # Format the diagnosis codes by adding punctuations
    mimic_diag = mimic_diag.with_columns(
        pl.col("diagnosis_codes").map_elements(mimic_utils.reformat_icd9cm_code, return_dtype=pl.Utf8)
    )
    mimic_proc = mimic_proc.with_columns(
        pl.col("procedure_codes").map_elements(mimic_utils.reformat_icd9pcs_code, return_dtype=pl.Utf8)
    )

    # Process codes and notes
    mimic_diag = parse_code_dataframe(
        mimic_diag,
        code_column="diagnosis_codes",
        code_type_column="diagnosis_code_type",
    )

    mimic_proc = parse_code_dataframe(
        mimic_proc,
        code_column="procedure_codes",
        code_type_column="procedure_code_type",
    )
    mimic_notes = parse_notes_dataframe(mimic_notes)
    mimic_codes = mimic_diag.join(mimic_proc, on=mimic_utils.ID_COLUMN, how="full", coalesce=True)

    mimic_codes = mimic_codes.with_columns(
        pl.concat_list([pl.col("diagnosis_codes").fill_null([]), pl.col("procedure_codes").fill_null([])])
        .map_elements(lambda x: list(set(x)), return_dtype=pl.List(pl.Utf8))  # Ensure uniqueness
        .alias("codes")
    )

    mimiciii = mimic_notes.join(mimic_codes, on=mimic_utils.ID_COLUMN, how="inner")

    # remove rare codes
    mimiciii_clean = mimic_utils.remove_rare_codes(mimiciii, ["codes"], 10)
    # mimiciii_50 = mimic_utils.keep_top_k_codes(mimiciii_clean, ["codes"], 50).filter(
    #     pl.col("codes").map_elements(len) > 0
    # )
    mimiciii_50 = mimiciii.with_columns(
        pl.col("codes")
        .map_elements(lambda x: [code for code in x if code in TOP_50_MULLENBACH_CODES], return_dtype=pl.List(pl.Utf8))
        .alias("codes")
    ).filter(pl.col("codes").map_elements(len) > 0)
    # save files to disk
    logger.info(f"Saving the MIMIC-III dataset to {OUTPUT_DIR}")
    mimiciii.write_parquet(OUTPUT_DIR / "mimiciii_full.parquet")
    mimiciii_50.write_parquet(OUTPUT_DIR / "mimiciii_50.parquet")
    mimiciii_clean.write_parquet(OUTPUT_DIR / "mimiciii_clean.parquet")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
