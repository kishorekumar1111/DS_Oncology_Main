import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PERSONALIZED PRECISION MEDICINE FOR ONCOLOGY
# STAGE 01 - DATA ENGINEER
# ============================================================

print("=" * 70)
print("PERSONALIZED PRECISION ONCOLOGY - DATA ENGINEER")
print("=" * 70)


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = Path("data/personalized_precision_oncology_unclean_1000x20.csv")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

CLEAN_FILE = OUTPUT_DIR / "oncology_master_clean.csv"


# ------------------------------------------------------------
# 2. LOAD RAW DATA
# ------------------------------------------------------------

print("\n[1] Loading raw oncology dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Raw dataset shape: {df.shape}")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ------------------------------------------------------------
# 3. STANDARDIZE COLUMN NAMES
# ------------------------------------------------------------

print("\n[2] Standardizing column names...")

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

print("Column names standardized.")


# ------------------------------------------------------------
# 4. REMOVE DUPLICATE PATIENT RECORDS
# ------------------------------------------------------------

print("\n[3] Checking duplicate patient records...")

duplicate_count = df.duplicated(subset=["Patient_ID"]).sum()

print(f"Duplicate Patient_ID records found: {duplicate_count}")

df = df.drop_duplicates(
    subset=["Patient_ID"],
    keep="first"
)

print(f"Rows after duplicate removal: {len(df)}")


# ------------------------------------------------------------
# 5. REMOVE EXTRA WHITESPACE FROM TEXT
# ------------------------------------------------------------

print("\n[4] Cleaning text whitespace...")

text_columns = df.select_dtypes(include=["object"]).columns

for column in text_columns:
    df[column] = df[column].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )

print("Whitespace cleaning completed.")


# ------------------------------------------------------------
# 6. STANDARDIZE SEX VALUES
# ------------------------------------------------------------

print("\n[5] Standardizing Sex values...")

if "Sex" in df.columns:

    df["Sex"] = (
        df["Sex"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({
            "m": "Male",
            "male": "Male",
            "f": "Female",
            "female": "Female",
            "": pd.NA,
            "nan": pd.NA
        })
    )

print(df["Sex"].value_counts(dropna=False))


# ------------------------------------------------------------
# 7. STANDARDIZE CANCER TYPE
# ------------------------------------------------------------

print("\n[6] Standardizing Cancer Type...")

if "Cancer_Type" in df.columns:

    df["Cancer_Type"] = (
        df["Cancer_Type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    cancer_mapping = {
        "nsclc": "NSCLC",
        "breast cancer": "Breast Cancer",
        "colorectal cancer": "Colorectal Cancer",
        "prostate cancer": "Prostate Cancer",
        "melanoma": "Melanoma",
        "pancreatic cancer": "Pancreatic Cancer",
        "ovarian cancer": "Ovarian Cancer",
        "gastric cancer": "Gastric Cancer",
        "liver cancer": "Liver Cancer",
        "renal cancer": "Renal Cancer"
    }

    df["Cancer_Type"] = df["Cancer_Type"].replace(cancer_mapping)


# ------------------------------------------------------------
# 8. STANDARDIZE GENE MUTATIONS
# ------------------------------------------------------------

print("\n[7] Standardizing genomic mutation values...")

if "Gene_Mutation" in df.columns:

    df["Gene_Mutation"] = (
        df["Gene_Mutation"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    gene_mapping = {
        "EGFR": "EGFR",
        "KRAS": "KRAS",
        "ALK": "ALK",
        "ROS1": "ROS1",
        "BRAF": "BRAF",
        "HER2": "HER2",
        "BRCA1": "BRCA1",
        "BRCA2": "BRCA2",
        "TP-53": "TP53",
        "TP53": "TP53",
        "PIK3CA": "PIK3CA",
        "MET": "MET",
        "RET": "RET",
        "NTRK": "NTRK",
        "PD-L1": "PD-L1",
        "NONE": "None",
        "UNKNOWN": "Unknown"
    }

    df["Gene_Mutation"] = df["Gene_Mutation"].replace(gene_mapping)


# ------------------------------------------------------------
# 9. STANDARDIZE CANCER STAGE
# ------------------------------------------------------------

print("\n[8] Standardizing Cancer Stage...")

if "Cancer_Stage" in df.columns:

    df["Cancer_Stage"] = (
        df["Cancer_Stage"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    stage_mapping = {
        "1": "I",
        "2": "II",
        "3": "III",
        "STAGE III": "III",
        "4": "IV",
        "STAGE IV": "IV"
    }

    df["Cancer_Stage"] = df["Cancer_Stage"].replace(stage_mapping)


# ------------------------------------------------------------
# 10. CONVERT NUMERIC COLUMNS
# ------------------------------------------------------------

print("\n[9] Converting numeric clinical fields...")

numeric_columns = [
    "Age",
    "ctDNA_Level",
    "Protein_Biomarker",
    "SpO2",
    "Heart_Rate",
    "WBC_Count",
    "Platelet_Count",
    "Creatinine",
    "Bilirubin",
    "BMI",
    "ECOG_Status",
    "Dose_mg"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

print("Numeric conversion completed.")


# ------------------------------------------------------------
# 11. HANDLE INVALID CLINICAL VALUES
# ------------------------------------------------------------

print("\n[10] Checking invalid clinical values...")


# Age
if "Age" in df.columns:

    df.loc[
        (df["Age"] < 18) | (df["Age"] > 100),
        "Age"
    ] = np.nan


# SpO2
if "SpO2" in df.columns:

    df.loc[
        (df["SpO2"] < 50) | (df["SpO2"] > 100),
        "SpO2"
    ] = np.nan


# BMI
if "BMI" in df.columns:

    df.loc[
        (df["BMI"] < 10) | (df["BMI"] > 60),
        "BMI"
    ] = np.nan


# Dose
if "Dose_mg" in df.columns:

    df.loc[
        (df["Dose_mg"] <= 0) | (df["Dose_mg"] > 1000),
        "Dose_mg"
    ] = np.nan


# ECOG
if "ECOG_Status" in df.columns:

    df.loc[
        (df["ECOG_Status"] < 0) |
        (df["ECOG_Status"] > 5),
        "ECOG_Status"
    ] = np.nan


# ------------------------------------------------------------
# 12. STANDARDIZE TREATMENT RESPONSE
# ------------------------------------------------------------

print("\n[11] Standardizing treatment response...")

if "Treatment_Response" in df.columns:

    df["Treatment_Response"] = (
        df["Treatment_Response"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    response_mapping = {

        "complete response": "Complete Response",
        "cr": "Complete Response",

        "partial response": "Partial Response",
        "pr": "Partial Response",

        "stable disease": "Stable Disease",
        "sd": "Stable Disease",
        "stable": "Stable Disease",

        "progressive disease": "Progressive Disease",
        "pd": "Progressive Disease",
        "progressive": "Progressive Disease"
    }

    df["Treatment_Response"] = (
        df["Treatment_Response"]
        .replace(response_mapping)
    )


# ------------------------------------------------------------
# 13. STANDARDIZE TOXICITY RISK
# ------------------------------------------------------------

print("\n[12] Standardizing toxicity risk...")

if "Toxicity_Risk" in df.columns:

    df["Toxicity_Risk"] = (
        df["Toxicity_Risk"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    toxicity_mapping = {

        "low": "Low",

        "moderate": "Moderate",

        "high": "High",
        "high risk": "High",
        "h": "High"
    }

    df["Toxicity_Risk"] = (
        df["Toxicity_Risk"]
        .replace(toxicity_mapping)
    )


# ------------------------------------------------------------
# 14. HANDLE MISSING VALUES
# ------------------------------------------------------------

print("\n[13] Handling missing values...")


# Numeric columns -> median
for column in numeric_columns:

    if column in df.columns:

        median_value = df[column].median()

        df[column] = df[column].fillna(median_value)


# Categorical columns -> Unknown
categorical_columns = [
    "Sex",
    "Cancer_Type",
    "Cancer_Stage",
    "Gene_Mutation",
    "Treatment",
    "Treatment_Response"
]

for column in categorical_columns:

    if column in df.columns:

        df[column] = df[column].fillna("Unknown")


# ------------------------------------------------------------
# 15. CREATE TIMESTAMP
# ------------------------------------------------------------

print("\n[14] Creating timestamp for real-time pipeline...")

df["Record_Timestamp"] = pd.date_range(
    start="2026-01-01",
    periods=len(df),
    freq="h"
)

# Move timestamp near Patient_ID
column_order = ["Patient_ID", "Record_Timestamp"] + [
    col for col in df.columns
    if col not in ["Patient_ID", "Record_Timestamp"]
]

df = df[column_order]


# ------------------------------------------------------------
# 16. FINAL DATA QUALITY CHECK
# ------------------------------------------------------------

print("\n[15] Final data quality check...")

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

print("\nFinal dataset shape:")
print(df.shape)


# ------------------------------------------------------------
# 17. SAVE CLEAN MASTER DATASET
# ------------------------------------------------------------

df.to_csv(
    CLEAN_FILE,
    index=False
)

print("\n" + "=" * 70)
print("DATA ENGINEER PIPELINE COMPLETED")
print("=" * 70)

print(f"\nClean master dataset saved to:")
print(CLEAN_FILE)

print(f"\nFinal rows    : {df.shape[0]}")
print(f"Final columns : {df.shape[1]}")

print("\nFirst 5 cleaned records:")
print(df.head().to_string(index=False))

print("\nData Engineer stage completed successfully.")