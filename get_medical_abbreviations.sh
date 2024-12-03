#!/bin/bash

# Enable exit on error and print commands as they execute
set -ex

# Step 1: Download the ZIP file
curl -sSL https://github.com/imantsm/medical_abbreviations/archive/4f1f4afefe0efc4b37fed43c8482d49ad87b0706.zip -o medical_abbreviations.zip

# Step 2: Extract the ZIP file
unzip -q medical_abbreviations.zip

# Step 3: Navigate to the CSVs directory
cd medical_abbreviations-4f1f4afefe0efc4b37fed43c8482d49ad87b0706/CSVs

# Step 4: Merge all CSVs into a single file
awk '(NR == 1) || (FNR > 1)' *.csv > medical_abbreviations.csv

# Step 5: Move the merged file to Downloads
mv medical_abbreviations.csv ~/Downloads

# Step 6: Clean up temporary files
cd ../../
rm -fr medical_abbreviations.zip medical_abbreviations-4f1f4afefe0efc4b37fed43c8482d49ad87b0706

# Print completion message
echo "Merged medical abbreviations CSV saved to ~/Downloads/medical_abbreviations.csv"
