"""
NOTE: This script is modified from https://github.com/JoakimEdin/explainable-medical-coding/blob/main/explainable_medical_coding/data/prepare_mimiciv.py

This script takes the raw data from the MIMIC-IV dataset and prepares it for the downstream COP-IV task.
The script does the following:
1. Loads the data from the csv files.
2. Renames the columns to match the column names in the MIMIC-IV dataset.
3. Adds punctuations to the ICD-9-CM. ICD-9-PCS, and ICD-10-CM codes (not needed for ICD-10-PCS codes).
4. Removes duplicate rows.
5. Removes cases with no codes.
6. Extracts the COP-IV tasks; Categorical LoS, mortality, careunit, and diagnoses/procedure.
7. Saves the data as parquet files.


"""

import logging
import random
from pathlib import Path

import polars as pl
from dotenv import find_dotenv, load_dotenv

from dataloader import mimic_utils

random.seed(10)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path("data/cop-iv/processed")


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
    df = df.filter(df[mimic_utils.TEXT_COLUMN].is_not_null())
    df = df.unique(subset=[mimic_utils.ID_COLUMN, mimic_utils.TEXT_COLUMN])
    return df


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info("Preparing the COP-IV dataset from raw data")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the dataframes
    mimic_notes = pl.read_csv(PROJECT_ROOT / "data/mimic-iv-note/raw/discharge.csv.gz")
    mimic_diag = pl.read_csv(
        PROJECT_ROOT / "data/mimic-iv/raw/diagnoses_icd.csv.gz",
        schema={
            "hadm_id": pl.Int64,
            "subject_id": pl.Int64,
            "icd_code": pl.String,
            "icd_version": pl.String,
        },
        truncate_ragged_lines=True,
    )
    mimic_proc = pl.read_csv(
        PROJECT_ROOT / "data/mimic-iv/raw/procedures_icd.csv.gz",
        schema={
            "hadm_id": pl.Int64,
            "subject_id": pl.Int64,
            "icd_code": pl.String,
            "icd_version": pl.String,
        },
        truncate_ragged_lines=True,
    )
    mimic_admissions = pl.read_csv(PROJECT_ROOT / "data/mimic-iv/raw/admissions.csv.gz")
    mimic_admissions = mimic_admissions.filter(
        mimic_admissions["los_days"].is_not_null() & mimic_admissions["hospital_expire_flag"].is_not_null()
    )

    # rename the columns
    mimic_notes = mimic_notes.rename(
        {
            "hadm_id": mimic_utils.ID_COLUMN,
            "subject_id": mimic_utils.SUBJECT_ID_COLUMN,
            "text": mimic_utils.TEXT_COLUMN,
        }
    )
    mimic_diag = mimic_diag.rename(
        {
            "hadm_id": mimic_utils.ID_COLUMN,
            "icd_code": "diagnosis_codes",
            "icd_version": "diagnosis_code_type",
        }
    ).drop(["subject_id"])
    mimic_proc = mimic_proc.rename(
        {
            "hadm_id": mimic_utils.ID_COLUMN,
            "icd_code": "procedure_codes",
            "icd_version": "procedure_code_type",
        }
    ).drop(["subject_id"])

    # Format the code type columns
    mimic_diag = mimic_diag.with_columns(mimic_diag["diagnosis_code_type"].cast(pl.Utf8))
    mimic_diag = mimic_diag.with_columns(mimic_diag["diagnosis_code_type"].str.replace("10", "icd10cm"))
    mimic_diag = mimic_diag.with_columns(mimic_diag["diagnosis_code_type"].str.replace("9", "icd9cm"))

    mimic_proc = mimic_proc.with_columns(mimic_proc["procedure_code_type"].cast(pl.Utf8))
    mimic_proc = mimic_proc.with_columns(mimic_proc["procedure_code_type"].str.replace("10", "icd10pcs"))
    mimic_proc = mimic_proc.with_columns(mimic_proc["procedure_code_type"].str.replace("9", "icd9pcs"))

    # Format the diagnosis codes by adding punctuation points
    formatted_codes = (
        pl.when(mimic_diag["diagnosis_code_type"] == "icd10cm")
        .then(mimic_diag["diagnosis_codes"].map_elements(mimic_utils.reformat_icd10cm_code, return_dtype=pl.Utf8))
        .otherwise(mimic_diag["diagnosis_codes"].map_elements(mimic_utils.reformat_icd9cm_code, return_dtype=pl.Utf8))
    )
    mimic_diag = mimic_diag.with_columns(formatted_codes)

    # Format the procedure codes by adding punctuation points
    formatted_codes = (
        pl.when(mimic_proc["procedure_code_type"] == "icd10pcs")
        .then(mimic_proc["procedure_codes"])
        .otherwise(mimic_proc["procedure_codes"].map_elements(mimic_utils.reformat_icd9pcs_code, return_dtype=pl.Utf8))
    )
    mimic_proc = mimic_proc.with_columns(formatted_codes)

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

    # Include the COP-IV tasks; Categorical LoS, mortality, careunit, and diagnoses/procedure
    mimic_diag = mimic_diag.with_columns(pl.col("diagnosis_codes").str.slice(3).alias("diagnosis"))
    mimic_proc = mimic_proc.with_columns(pl.col("procedure_codes").str.slice(4).alias("procedure"))
    mimic_codes = mimic_diag.join(mimic_proc, on=mimic_utils.ID_COLUMN, how="full", coalesce=True)

    mimic_admissions = mimic_admissions.with_columns(
        mimic_admissions["los_days"].map(mimic_utils.map_los_to_class).alias("los_categorical")
    )
    mimic_admissions = mimic_admissions.with_columns(
        mimic_admissions["hospital_expire_flag"].cast(pl.Int8).alias("mortality")
    )
    mimic_admissions = mimic_admissions.with_columns(
        mimic_admissions["careunit"].map(mimic_utils.map_careunit).alias("careunit")
    )

    # save files to disk
    logger.info(f"Saving the COP-IV dataset to {OUTPUT_DIR}")
    mimic_notes = parse_notes_dataframe(mimic_notes)
    copiv = mimic_notes.join(mimic_codes, on=mimic_utils.ID_COLUMN, how="inner")
    copiv = copiv.join(mimic_admissions[["hadm_id", "los_categorical", "mortality", "careunit"]], on="hadm_id", how="inner")
    copiv = copiv.with_columns(copiv["note_type"].str.replace("DS", "discharge_summary"))
    copiv.write_parquet(OUTPUT_DIR / "copiv.parquet")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
