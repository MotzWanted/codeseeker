from pathlib import Path
import polars as pl
import logging
from loguru import logger

from dataloader import mimic_utils

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path("data/meddec/processed")

# Define paths for the input data
MIMIC_NOTES_PATH = PROJECT_ROOT / "data/mimic-iii/processed"
MEDDEC_PATH = PROJECT_ROOT / "data/meddec/raw"


def load_mimic_notes() -> pl.DataFrame:
    """
    Load MIMIC-III notes from the Parquet files.
    """
    logger.info(f"Loading MIMIC-III notes from {MIMIC_NOTES_PATH}")
    mimic_notes = pl.read_parquet(MIMIC_NOTES_PATH / "mimiciii.parquet")

    # Clean and preprocess `note_id` to handle multi-part IDs
    mimic_notes = mimic_notes.with_columns(
        pl.col("note_id")
        .str.replace(r"[^\d]", "")  # Remove non-numeric characters
        .cast(pl.Int64, strict=False)  # Allow nulls for invalid conversions
    )

    # Drop rows where `note_id` is invalid after cleaning
    mimic_notes = mimic_notes.filter(pl.col("note_id").is_not_null())
    return mimic_notes[
        mimic_utils.SUBJECT_ID_COLUMN, mimic_utils.ID_COLUMN, mimic_utils.ROW_ID_COLUMN, mimic_utils.TEXT_COLUMN
    ]


def transform_phenotypes(df: pl.DataFrame) -> pl.DataFrame:
    # Loop through all columns to create a single "PHENOTYPE" column
    phenotype_exprs = []

    for col in df.columns:
        if col in [mimic_utils.ID_COLUMN, mimic_utils.SUBJECT_ID_COLUMN, mimic_utils.ROW_ID_COLUMN, "OPERATOR"]:
            continue

        expr = pl.when(pl.col(col) == 1).then(pl.lit(col))
        phenotype_exprs.append(expr)

    # Create a new dataframe with the "PHENOTYPE" column
    return df.select(
        [
            *[
                pl.col(col)
                for col in [mimic_utils.ID_COLUMN, mimic_utils.SUBJECT_ID_COLUMN, mimic_utils.ROW_ID_COLUMN, "OPERATOR"]
                if col in df.columns
            ],
            pl.coalesce(phenotype_exprs).alias("PHENOTYPE"),
        ]
    )


def load_phenotypes() -> pl.DataFrame:
    """
    Load phenotype annotations from the CSV file.
    """
    logger.info(f"Loading phenotypes from {MEDDEC_PATH}")
    phenotypes = pl.read_csv(MEDDEC_PATH / "ACTdb102003.csv")
    phenotypes = phenotypes.rename(
        {
            "SUBJECT_ID": mimic_utils.SUBJECT_ID_COLUMN,
            "HADM_ID": mimic_utils.ID_COLUMN,
            "ROW_ID": mimic_utils.ROW_ID_COLUMN,
        }
    ).drop("BATCH.ID")
    return transform_phenotypes(phenotypes)


def load_meddec_annotations() -> pl.DataFrame:
    """
    Load MedDec annotations from JSON files.
    """
    logger.info(f"Loading MedDec annotations from {MEDDEC_PATH}")
    json_files = list(MEDDEC_PATH.glob("*.json"))
    meddec_frames = []
    for file in json_files:
        df = pl.read_json(file)
        annotations = df.explode("annotations")
        annotations = annotations.with_columns(
            [
                pl.col("annotations").struct.field("start_offset").cast(pl.Int64).alias("start_offset"),
                pl.col("annotations").struct.field("end_offset").cast(pl.Int64).alias("end_offset"),
                pl.col("annotations").struct.field("category").alias("category"),
                pl.col("annotations").struct.field("decision").alias("decision"),
                pl.col("annotations").struct.field("annotation_id").alias("annotation_id"),
            ]
        ).drop("annotations")
        annotations = annotations.with_columns(pl.lit(file.name).alias("file_name"))
        meddec_frames.append(annotations)
    return pl.concat(meddec_frames, rechunk=True)


def merge_data(mimic_notes: pl.DataFrame, phenotypes: pl.DataFrame, meddec_annotations: pl.DataFrame) -> pl.DataFrame:
    """
    Merge MedDec annotations with MIMIC-III notes and Phenotypes based on common identifiers.
    """
    logger.info("Merging MedDec annotations with MIMIC-III notes and phenotypes")
    # Extract identifiers from the discharge summary ID and convert to the appropriate data types
    # Extract SUBJECT_ID
    meddec_annotations = meddec_annotations.with_columns(
        meddec_annotations["discharge_summary_id"]
        .str.extract(r"^(\d+)_", 1)
        .cast(pl.Int64)
        .alias(mimic_utils.SUBJECT_ID_COLUMN)
    )

    # Extract HADM_ID
    meddec_annotations = meddec_annotations.with_columns(
        meddec_annotations["discharge_summary_id"]
        .str.extract(r"_(\d+)_", 1)
        .cast(pl.Int64)
        .alias(mimic_utils.ID_COLUMN)
    )

    # Extract ROW_ID
    meddec_annotations = meddec_annotations.with_columns(
        meddec_annotations["discharge_summary_id"]
        .str.extract(r"_(\d+)?_{0,2}$", 1)
        .cast(pl.Int64)
        .alias(mimic_utils.ROW_ID_COLUMN)
    )

    mimic_notes = mimic_notes.with_columns(
        [
            pl.col(mimic_utils.SUBJECT_ID_COLUMN).cast(pl.Int64),
            pl.col(mimic_utils.ID_COLUMN).cast(pl.Int64),
            pl.col(mimic_utils.ROW_ID_COLUMN).cast(pl.Int64),
        ]
    )

    # Ensure phenotypes join keys have matching data types
    phenotypes = phenotypes.with_columns([pl.col(mimic_utils.SUBJECT_ID_COLUMN).cast(pl.Int64)])

    # Join MedDec annotations with MIMIC-III notes
    merged_data = meddec_annotations.join(
        mimic_notes, on=[mimic_utils.SUBJECT_ID_COLUMN, mimic_utils.ID_COLUMN], how="left"
    )

    # Join the resulting data with phenotypes
    # NOTE: merging with phenotypes is currently removing half of the annotations. Investigate this further.
    # merged_data = merged_data.join(
    #     phenotypes, on=[mimic_utils.SUBJECT_ID_COLUMN, mimic_utils.ID_COLUMN, mimic_utils.ROW_ID_COLUMN], how="left"
    # )

    return merged_data.drop(["discharge_summary_id"])


def main():
    """Runs data processing scripts to turn raw data from (..data/meddec/raw) into a parquet file."""

    logger.info(f"Saving the MedDec dataset to {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the data
    mimic_notes = load_mimic_notes()
    phenotypes = load_phenotypes()
    meddec_annotations = load_meddec_annotations()

    # Merge the data
    merged_data = merge_data(mimic_notes, phenotypes, meddec_annotations)

    # Write merged data to Parquet
    merged_data.write_parquet(OUTPUT_DIR / "meddec.parquet")
    logger.info(f"MedDec dataset saved successfully to {OUTPUT_DIR / 'meddec.parquet'}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
