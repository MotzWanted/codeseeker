from hashlib import sha512
import polars as pl
from dataloader.constants import PROJECT_ROOT

pl.Config.set_tbl_cols(100)  # Set max columns displayed
pl.Config.set_fmt_str_lengths(200)  # Set max string length before truncation


mimic_notes = (
    pl.read_csv(PROJECT_ROOT / "data/mimic-iii/raw/NOTEEVENTS.csv.gz")
    .filter(pl.col("HADM_ID").is_not_null() & pl.col("CGID").is_not_null())
    .drop(["ISERROR", "STORETIME"])
).with_columns(
    pl.col("TEXT").map_elements(lambda x: sha512(x.encode("utf-8")).hexdigest(), return_dtype=pl.Utf8).alias("HASH")
)
hash_with_multiple_cgids = (
    mimic_notes.group_by("HASH")
    .agg(pl.col("CGID").n_unique().alias("unique_cgid_count"))
    .filter(pl.col("unique_cgid_count") > 1)  # Keep only those with multiple CGIDs
)
mimic_notes_deduplicated = mimic_notes.filter(~pl.col("HASH").is_in(hash_with_multiple_cgids["HASH"]))
caregivers = pl.read_csv(
    PROJECT_ROOT / "data/mimic-iii/raw/CAREGIVERS.csv.gz",
    schema={
        "ROW_ID": pl.UInt32,
        "CGID": pl.Utf8,
        "LABEL": pl.Utf8,
        "DESCRIPTION": pl.Utf8,
    },
).drop(["ROW_ID", "LABEL"])

multi_notes = (
    mimic_notes_deduplicated.group_by(["SUBJECT_ID", "HADM_ID"])
    .agg(
        [
            pl.col("ROW_ID").n_unique().alias("NUMBER_OF_ROWS"),
            pl.col("CGID").n_unique().alias("NUMBER_OF_CAREGIVERS"),
            pl.col("CATEGORY").n_unique().alias("NUMBER_OF_CATEGORIES"),
            pl.col("CATEGORY").unique().alias("CATEGORIES_LIST"),  # Collect unique categories
            pl.col("HASH").n_unique().alias("HASH_COUNT"),  # Collect unique hashes
        ]
    )
    .sort("NUMBER_OF_CATEGORIES", descending=True)
    .filter(pl.col("NUMBER_OF_CATEGORIES") > 3)
)

filtered_mimic_notes = mimic_notes_deduplicated.filter(
    pl.col("HADM_ID").is_in(multi_notes.select("HADM_ID").to_series())
).sort(by=["SUBJECT_ID", "HADM_ID", "CHARTDATE"])

multi_mimic_notes = (
    filtered_mimic_notes.join(caregivers, on=["CGID"], how="left")
    .rename({"DESCRIPTION_right": "CGID_DESCRIPTION"})
    .select(
        [
            "ROW_ID",
            "SUBJECT_ID",
            "HADM_ID",
            "CHARTDATE",
            "CHARTTIME",
            "CATEGORY",
            "DESCRIPTION",
            "CGID",
            "CGID_DESCRIPTION",
            "TEXT",
        ]
    )
)
multi_mimic_notes.write_parquet("/Users/amo/Downloads/multi_mimic_notes.parquet")
multi_mimic_notes.write_csv("/Users/amo/Downloads/multi_mimic_notes.csv")

patients = (
    multi_mimic_notes.unique(subset="SUBJECT_ID").select("SUBJECT_ID").sample(fraction=0.1)
)  # Adjust the fraction as needed

subset_multi_mimic_notes = multi_mimic_notes.filter(pl.col("SUBJECT_ID").is_in(patients.to_series()))
subset_multi_mimic_notes.write_csv("/Users/amo/Downloads/subset_multi_mimic_notes.csv")
subset_multi_mimic_notes = multi_mimic_notes.sample(fraction=0.01)  # Adjust the fraction as needed
subset_multi_mimic_notes.write_parquet("/Users/amo/Downloads/subset_multi_mimic_notes.parquet")
subset_multi_mimic_notes.write_csv("/Users/amo/Downloads/subset_multi_mimic_notes.csv")


subset_multi_mimic_notes.filter(pl.col("HADM_ID") == 111578)


subset_multi_mimic_notes.group_by(["SUBJECT_ID", "HADM_ID"]).agg(
    [
        pl.col("ROW_ID").n_unique().alias("NUMBER_OF_ROWS"),
        pl.col("CGID").n_unique().alias("NUMBER_OF_CAREGIVERS"),
        pl.col("CATEGORY").n_unique().alias("NUMBER_OF_CATEGORIES"),
        pl.col("CATEGORY").unique().alias("CATEGORIES_LIST"),  # Collect unique categories
    ]
).sort("NUMBER_OF_CATEGORIES", descending=True)


sepsis_admission_notes = (
    multi_mimic_notes.group_by(["SUBJECT_ID", "HADM_ID"])
    .agg(pl.col("TEXT"))
    .filter(pl.col("TEXT").map_elements(lambda s: sum("sepsis" in x.lower() for x in s) > len(s) * 0.7))
)


sepsis_notes = multi_mimic_notes.filter(
    pl.col("HADM_ID").is_in(sepsis_admission_notes.select("HADM_ID").to_series())
).sort(by=["SUBJECT_ID", "HADM_ID", "CHARTDATE"])


multi_sepsis_notes = (
    sepsis_notes.group_by(["SUBJECT_ID", "HADM_ID"])
    .agg(
        [
            pl.col("ROW_ID").n_unique().alias("NUMBER_OF_ROWS"),
            pl.col("CGID").n_unique().alias("NUMBER_OF_CAREGIVERS"),
            pl.col("CATEGORY").n_unique().alias("NUMBER_OF_CATEGORIES"),
            pl.col("CATEGORY").unique().alias("CATEGORIES_LIST"),  # Collect unique categories
        ]
    )
    .sort("NUMBER_OF_CATEGORIES", descending=True)
    .filter(pl.col("NUMBER_OF_CATEGORIES") > 3)
)

filtered_sepsis_notes = multi_mimic_notes.filter(
    pl.col("HADM_ID").is_in(multi_sepsis_notes.select("HADM_ID").to_series())
).sort(by=["SUBJECT_ID", "HADM_ID", "CHARTDATE"])

filtered_sepsis_notes.group_by(["SUBJECT_ID", "HADM_ID"]).agg(
    [
        pl.col("ROW_ID").n_unique().alias("NUMBER_OF_ROWS"),
        pl.col("CGID").n_unique().alias("NUMBER_OF_CAREGIVERS"),
        pl.col("CATEGORY").n_unique().alias("NUMBER_OF_CATEGORIES"),
        pl.col("CATEGORY").unique().alias("CATEGORIES_LIST"),  # Collect unique categories
    ]
).sort("NUMBER_OF_CATEGORIES", descending=True)
filtered_sepsis_notes.write_csv("/Users/amo/Downloads/filtered_sepsis_notes.csv")
