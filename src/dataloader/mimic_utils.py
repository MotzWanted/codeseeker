"""Modified version of the data_helper_functions.py file from the https://github.com/JoakimEdin/explainable-medical-coding/blob/main/explainable_medical_coding/utils/data_helper_functions.py."""

import numpy as np
import polars as pl
from datasets import DatasetDict
from typing import Literal

ID_COLUMN = "_id"
TEXT_COLUMN = "text"
TEXT_ADMISSION_COLUMN = "text_admission"
TARGET_COLUMN = "target"
SUBJECT_ID_COLUMN = "subject_id"
SOS = "<sos>"
EOS = "<eos>"
PAD = "<pad>"


def is_list_empty(x: list) -> bool:
    """Check if a nested list is empty.

    Args:
        x (list): The list to check.

    Returns:
        bool: Whether the list is empty.
    """
    if isinstance(x, list):  # Is a list
        return all(map(is_list_empty, x))
    return False  # Not a list


def reformat_icd9cm_code(code: str) -> str:
    """
    Adds a punctuation in a ICD-9-CM code that is without punctuations.
    Before: 0019
    After:  001.9
    """
    if "." in code:
        return code

    if code.startswith("E"):
        if len(code) > 4:
            return code[:4] + "." + code[4:]
    elif len(code) > 3:
        return code[:3] + "." + code[3:]
    return code


def reformat_icd9pcs_code(code: str) -> str:
    """
    Adds a punctuation in a ICD-9-PCS code that is without punctuations.

    Before: 0019
    After:  00.19
    """

    if "." in code:
        return code
    if len(code) > 2:
        return code[:2] + "." + code[2:]
    return code


def reformat_icd10cm_code(code: str) -> str:
    """
    Adds a punctuation in a ICD-10-CM code that is without punctuations.
    Before: A019
    After:  A01.9
    """
    if len(code) > 3:
        return code[:3] + "." + code[3:]
    else:
        return code


def remove_rare_codes(df: pl.DataFrame, code_columns: list[str], min_count: int) -> pl.DataFrame:
    """Removes codes that appear less than min_count times in the dataframe.

    Args:
        df (pl.DataFrame): dataframe with codes
        code_columns (list[str]): list of columns with codes
        min_count (int): minimum number of times a code has to appear in the dataframe to be kept

    Returns:
        pl.DataFrame: dataframe with codes that appear more than min_count times
    """
    for code_column in code_columns:
        code_exploded = df[["_id", code_column]].explode(code_column)
        code_counts = code_exploded[code_column].value_counts()
        codes_to_include = set(code_counts.filter(code_counts["count"] >= min_count)[code_column])
        code_exploded_filtered = code_exploded.filter(pl.col(code_column).is_in(codes_to_include))
        code_filtered = code_exploded_filtered.group_by("_id").agg(pl.col(code_column))
        df = df.drop(code_column)
        df = df.join(code_filtered, on="_id", how="left")
    return df


def keep_top_k_codes(df: pl.DataFrame, code_columns: list[str], k: int) -> pl.DataFrame:
    """Only keep the k most common codes.

    Args:
        df (pl.DataFrame): dataframe with codes
        code_columns (list[str]): list of columns with codes
        k (int): Number of codes to keep

    Returns:
        pl.DataFrame: dataframe with k number of codes
    """
    code_counts = None

    # Iterate over each code column to get the counts
    for code_column in code_columns:
        code_exploded = df[["_id", code_column]].explode(code_column)
        if code_counts is None:
            code_counts = code_exploded[code_column].value_counts().rename({code_column: "codes"})
        else:
            code_counts.extend(code_exploded[code_column].value_counts().rename({code_column: "codes"}))

    # Ensure code_counts is properly initialized and filtered before proceeding
    if code_counts is None:
        raise ValueError("No code columns provided or no valid codes found in the DataFrame.")

    # Remove null values from code counts
    code_counts_filter: pl.DataFrame = code_counts.filter(pl.col("codes").is_not_null())

    # Get the top k codes
    codes_to_include = set(code_counts_filter.sort("count", descending=True)[:k]["codes"])

    # Filter the original DataFrame to keep only the top k codes
    for code_column in code_columns:
        code_exploded = df[["_id", code_column]].explode(code_column)
        code_exploded_filtered = code_exploded.filter(pl.col(code_column).is_in(codes_to_include))
        code_filtered = code_exploded_filtered.group_by("_id").agg(pl.col(code_column))
        df = df.drop(code_column)
        df = df.join(code_filtered, on="_id", how="left")

    return df


def create_targets_column(example: dict, target_columns: list[str]) -> dict[str, list[str]]:
    """Create the targets column by combining the columns specified in target_columns.

    Args:
        example (dict): The example.
        target_columns (list[str]): The target columns.

    Returns:
        dict[str, list[str]]: The example with the new targets column.
    """
    example[TARGET_COLUMN] = []
    for target_column in target_columns:
        if example[target_column] is not None:
            example[TARGET_COLUMN] += example[target_column]
    return example


def get_unique_targets(dataset: DatasetDict) -> list[str]:
    """Get unique targets from the dataset.

    Args:
        dataset (DatasetDict): The dataset.

    Returns:
        list[str]: The unique targets.
    """
    targets = []
    for _, split in dataset.with_format("pandas").items():
        targets.append(split[TARGET_COLUMN].explode().unique())
    return list(np.unique(np.concatenate(targets)))


def reformat_icd10cm_code_vectorized(codes: pl.Expr) -> pl.Expr:
    """
    Vectorized reformatting for ICD-10-CM codes using Polars expressions.
    Args:
        codes (pl.Expr): Polars expression for ICD-10-CM codes.
    Returns:
        pl.Expr: Reformatted ICD-10-CM codes.
    """
    return codes.str.slice(0, 3).str.concat("." + codes.str.slice(3))


def get_code2description_mimiciv(icd_version: int = 10) -> dict[str, str]:
    """Get a dictionary mapping ICD codes to descriptions.

    Args:
        icd_version (int): Version of the ICD code (9 or 10).

    Returns:
        dict[str, str]: Dictionary mapping ICD codes to descriptions.
    """

    if icd_version not in [9, 10]:
        raise ValueError("icd_version must be either 9 or 10")

    # Read the CSV file with Polars.
    df_descriptions = pl.read_csv(
        "data/raw/physionet.org/files/mimiciv/2.2/hosp/d_icd_diagnoses.csv.gz",
        schema={"icd_code": pl.Utf8, "icd_version": pl.Int32, "long_title": pl.Utf8},
    )

    # Filter rows where 'icd_version' matches the specified version.
    df_descriptions = df_descriptions.filter(pl.col("icd_version") == icd_version)

    # Rename columns.
    df_descriptions = df_descriptions.rename({"icd_code": "target"})

    # Apply the reformat_icd10cm_code_vectorized function using Polars expressions.
    df_descriptions = df_descriptions.with_columns(reformat_icd10cm_code_vectorized(pl.col("target")).alias("target"))

    # Create a dictionary from the 'target' and 'long_title' columns.
    code_to_description_dict = df_descriptions.select(["target", "long_title"]).to_dicts()

    # Return a dictionary mapping 'target' to 'long_title'.
    return {row["target"]: row["long_title"] for row in code_to_description_dict}


def reformat_icd9cm_code_vectorized(codes: pl.Expr) -> pl.Expr:
    """
    Vectorized reformatting for ICD-9-CM codes using Polars expressions.
    Args:
        codes (pl.Expr): Polars expression for ICD-9-CM codes.
    Returns:
        pl.Expr: Reformatted ICD-9-CM codes.
    """
    return (
        pl.when(codes.str.slice(0, 1) == "E")
        .then(codes.str.slice(0, 4).str.concat("." + codes.str.slice(4)))
        .otherwise(codes.str.slice(0, 3).str.concat("." + codes.str.slice(3)))
    )


def reformat_icd9pcs_code_vectorized(codes: pl.Expr) -> pl.Expr:
    """
    Vectorized reformatting for ICD-9-PCS codes using Polars expressions.
    Args:
        codes (pl.Expr): Polars expression for ICD-9-PCS codes.
    Returns:
        pl.Expr: Reformatted ICD-9-PCS codes.
    """
    return codes.str.slice(0, 2).str.concat("." + codes.str.slice(2))


def get_code2description_mimiciii() -> dict[str, str]:
    """Get a dictionary mapping ICD codes to descriptions.

    Returns:
        dict[str, str]: Dictionary mapping ICD codes to descriptions
    """
    # Read the CSV files with Polars.
    df_descriptions_diag = pl.read_csv(
        "data/raw/physionet.org/files/mimiciii/1.4/D_ICD_DIAGNOSES.csv.gz",
        schema={"ICD9_CODE": str},
    )

    df_descriptions_proc = pl.read_csv(
        "data/raw/physionet.org/files/mimiciii/1.4/D_ICD_PROCEDURES.csv.gz",
        schema={"ICD9_CODE": str},
    )

    # Rename columns.
    df_descriptions_diag = df_descriptions_diag.rename({"ICD9_CODE": "target", "LONG_TITLE": "long_title"})
    df_descriptions_proc = df_descriptions_proc.rename({"ICD9_CODE": "target", "LONG_TITLE": "long_title"})

    # Apply the reformat functions using Polars native operations.
    df_descriptions_diag = df_descriptions_diag.with_columns(
        reformat_icd9cm_code_vectorized(pl.col("target")).alias("target")
    )
    df_descriptions_proc = df_descriptions_proc.with_columns(
        reformat_icd9pcs_code_vectorized(pl.col("target")).alias("target")
    )

    # Concatenate the two DataFrames.
    df_descriptions = pl.concat([df_descriptions_diag, df_descriptions_proc])

    # Create a dictionary from the 'target' and 'long_title' columns.
    code_to_description_dict = df_descriptions.select(["target", "long_title"]).to_dicts()

    # Return a dictionary mapping 'target' to 'long_title'.
    return {row["target"]: row["long_title"] for row in code_to_description_dict}


def clean_empty_codes(example):
    if example["procedure_codes"] is None:
        example["procedure_codes"] = []
    else:
        example["procedure_codes"] = [c for c in example["procedure_codes"] if c]

    if example["diagnosis_codes"] is None:
        example["diagnosis_codes"] = []
    else:
        example["diagnosis_codes"] = [c for c in example["diagnosis_codes"] if c]
    return example


def join_text(example):
    example["text"] = " ".join(example["text"])
    return example


def filter_unknown_targets(example: dict, known_targets: set[str]) -> dict:
    """Filter out targets that are not in the target tokenizer.

    Args:
        example (dict): Example.
        target_tokenizer (TargetTokenizer): Target tokenizer.
        known_targets (set[str]): Known targets.

    Returns:
        dict: Example with filtered targets.
    """

    length_before = len(example[TARGET_COLUMN])
    example[TARGET_COLUMN] = [target for target in example[TARGET_COLUMN] if target in known_targets]

    if length_before == len(example[TARGET_COLUMN]):
        return example

    if "diagnosis_codes" in example:
        known_diagnosis_target_ids = [
            idx for idx, target in enumerate(example["diagnosis_codes"]) if target in known_targets
        ]
        example["diagnosis_codes"] = [example["diagnosis_codes"][idx] for idx in known_diagnosis_target_ids]
        if "diagnosis_code_spans" in example:
            if len(known_diagnosis_target_ids) > 0:
                example["diagnosis_code_spans"] = [
                    example["diagnosis_code_spans"][idx] for idx in known_diagnosis_target_ids
                ]

            else:
                example["diagnosis_code_spans"] = None

    if "procedure_codes" in example:
        known_procedure_target_ids = [
            idx for idx, target in enumerate(example["procedure_codes"]) if target in known_targets
        ]
        example["procedure_codes"] = [example["procedure_codes"][idx] for idx in known_procedure_target_ids]
        if "procedure_code_spans" in example:
            if len(known_procedure_target_ids) > 0:
                example["procedure_code_spans"] = [
                    example["procedure_code_spans"][idx] for idx in known_procedure_target_ids
                ]
            else:
                example["procedure_code_spans"] = None

    return example


def map_careunit(careunit: str) -> str:
    """
    Map raw care units to broader categories.

    Args:
        careunit (str): Raw care unit.

    Returns:
        str: Mapped care unit category.
    """
    surgery = ['Med/Surg', 'Surgery', 'Medical/Surgical (Gynecology)', 'PACU']
    cardiology = ['Cardiology', 'Medicine/Cardiology']
    neurology = ['Neurology', 'Neuro Intermediate']
    obstetrics = ['Labor & Delivery', 'Obstetrics Postpartum']
    oncology = ['Hematology/Oncology']
    observation = ['Observation']

    if careunit in surgery:
        return "surgery"
    elif careunit in cardiology:
        return "cardiology"
    elif careunit in neurology:
        return "neurology"
    elif careunit in obstetrics:
        return "obstetrics"
    elif careunit in oncology:
        return "oncology"
    elif careunit in observation:
        return "observation"
    else:
        return "other"

def filter_admission_text(notes_df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter text information by section and only keep sections that are known at admission time.
    Polars-compatible implementation.
    """
    # Define the sections and their corresponding regex patterns
    admission_sections = {
        "chief_complaint": r"chief complaint:",
        "present_illness": r"present illness:",
        "medical_history": r"medical history:",
        "medication_adm": r"medications on admission:",
        "allergies": r"allergies:",
        "physical_exam": r"physical exam:",
        "family_history": r"family history:",
        "social_history": r"social history:",
    }

    # Replace linebreak indicators
    notes_df = notes_df.with_columns(
        pl.col("text").str.replace("___\nFamily History:", "___\n\nFamily History:")
    )

    # Extract each section using regex
    for key, section in admission_sections.items():
        pattern = rf"{section}([\s\S]+?)\n\s*?\n[^(\\|\d|\.)]+?:"
        extracted_col = notes_df["text"].str.extract(pattern, group_index=1)
        # Clean up the extracted text
        extracted_col = extracted_col.str.replace("\n", " ").str.strip_chars().fill_null("")
        notes_df = notes_df.with_columns(pl.col(extracted_col).alias(key))

    # Filter out rows with missing main information
    filter_condition = (
        (pl.col("chief_complaint") != "") |
        (pl.col("present_illness") != "") |
        (pl.col("medical_history") != "")
    )
    notes_df = notes_df.filter(filter_condition)

    # Combine the sections into a single column with headers
    combined_text = (
        "CHIEF COMPLAINT: " + pl.col("chief_complaint") +
        "\n\nPRESENT ILLNESS: " + pl.col("present_illness") +
        "\n\nMEDICAL HISTORY: " + pl.col("medical_history") +
        "\n\nMEDICATION ON ADMISSION: " + pl.col("medication_adm") +
        "\n\nALLERGIES: " + pl.col("allergies") +
        "\n\nPHYSICAL EXAM: " + pl.col("physical_exam") +
        "\n\nFAMILY HISTORY: " + pl.col("family_history") +
        "\n\nSOCIAL HISTORY: " + pl.col("social_history")
    )
    notes_df = notes_df.with_columns(combined_text.alias(TEXT_ADMISSION_COLUMN))
    return notes_df
