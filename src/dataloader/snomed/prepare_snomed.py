from pathlib import Path

import polars as pl
import logging
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path("data/snomed/processed")
MEDICAL_CODING_SYSTEMS_DIR = PROJECT_ROOT / "data/medical-coding-systems"


def main():
    snomed_concepts = pl.read_csv(
        MEDICAL_CODING_SYSTEMS_DIR / "snomed/CONCEPT.csv",
        truncate_ragged_lines=True,
        separator="\t",
        quote_char=None,  # Disable special quote character processing
        infer_schema_length=10000,
        schema_overrides={"concept_name": pl.Utf8, "invalid_reason": pl.Utf8},
        ignore_errors=True,
    )

    # Filter to SNOMED concepts
    snomed_concepts = snomed_concepts.filter(pl.col("vocabulary_id") == "SNOMED")

    duplicated_concept_ids = (
        snomed_concepts.group_by("concept_code")
        .agg(pl.len())  # Use `pl.len()` to count occurrences
        .filter(pl.col("len") > 1)  # Keep only duplicates
    )
    logger.info(f"Found {duplicated_concept_ids.height} duplicated concept codes")

    # Exclude unused columns
    snomed_concepts = snomed_concepts["concept_code", "concept_name"]
    snomed_concepts = snomed_concepts.rename({"concept_code": "concept_id"})
    # Filter rows where `concept_id` contains only digits and convert to integer type
    snomed_concepts = snomed_concepts.filter(pl.col("concept_id").str.contains(r"^\d+$"))
    snomed_concepts = snomed_concepts.with_columns(pl.col("concept_id").cast(pl.Int64))

    notes = pl.read_csv(PROJECT_ROOT / "data/snomed/raw/mimic-iv_notes_training_set.csv")
    annotations = pl.read_csv(PROJECT_ROOT / "data/snomed/raw/train_annotations.csv")

    note_annotations = annotations.join(notes, on="note_id", how="inner")
    note_concept_annotations = note_annotations.join(snomed_concepts, on="concept_id", how="left")

    missing_concepts = note_concept_annotations.filter(pl.col("concept_name").is_null())
    logger.info(f"Found {missing_concepts.height} missing concepts")

    logger.info(f"Saving the SNOMED Entity Linking Challenge dataset to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    note_concept_annotations.write_parquet(OUTPUT_DIR / "snomed.parquet")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
