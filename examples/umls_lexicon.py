import csv
import pandas as pd
from pathlib import Path

# Path to your MRCONSO.RRF file
rrf_file_path = Path("~/Downloads/2024AA/META/MRCONSO.RRF")

# Read the .RRF file
df = pd.read_csv(
    rrf_file_path, sep="|", header=None, dtype=str, engine="python", quoting=csv.QUOTE_NONE, encoding_errors="ignore"
)
df = df.iloc[:, :-1]  # Remove the last empty column

# Assign column names
column_names = [
    "CUI",
    "LAT",
    "TS",
    "LUI",
    "STT",
    "SUI",
    "ISPREF",
    "AUI",
    "SAUI",
    "SCUI",
    "SDUI",
    "SAB",
    "TTY",
    "CODE",
    "STR",
    "SRL",
    "SUPPRESS",
    "CVF",
]
df.columns = column_names

# Example processing: Filter English preferred terms
eng_pref_terms = df[(df["LAT"] == "ENG") & (df["ISPREF"] == "Y")]

# Display the first few rows
print(eng_pref_terms.head())
