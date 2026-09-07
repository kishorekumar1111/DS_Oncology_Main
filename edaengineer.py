import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PERSONALIZED PRECISION MEDICINE FOR ONCOLOGY
# STAGE 01 - EDA ENGINEER
# ============================================================

print("=" * 70)
print("PERSONALIZED PRECISION ONCOLOGY - EDA ENGINEER")
print("=" * 70)


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

INPUT_FILE = Path("output/oncology_master_clean.csv")

EDA_DIR = Path("output/eda")
EDA_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD CLEAN MASTER DATASET
# ------------------------------------------------------------

print("\n[1] Loading clean master dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ------------------------------------------------------------
# 3. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\n[2] Dataset information")
print("-" * 50)

print(df.info())

print("\nFirst 5 records:")
print(df.head())

print("\nLast 5 records:")
print(df.tail())


# ------------------------------------------------------------
# 4. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n[3] Missing value analysis")
print("-" * 50)

missing_values = df.isnull().sum()

print(missing_values)

missing_report = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isnull().sum().values,
    "Missing_Percentage":
        (df.isnull().sum().values / len(df)) * 100
})

missing_report.to_csv(
    EDA_DIR / "missing_value_report.csv",
    index=False
)


# ------------------------------------------------------------
# 5. CHECK DUPLICATES
# ------------------------------------------------------------

print("\n[4] Duplicate analysis")
print("-" * 50)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count}")


# ------------------------------------------------------------
# 6. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n[5] Descriptive statistics")
print("-" * 50)

numeric_df = df.select_dtypes(include=np.number)

statistics = numeric_df.describe().T

print(statistics)

statistics.to_csv(
    EDA_DIR / "descriptive_statistics.csv"
)


# ------------------------------------------------------------
# 7. CATEGORICAL DATA DISTRIBUTION
# ------------------------------------------------------------

print("\n[6] Categorical distribution analysis")
print("-" * 50)

categorical_columns = [
    "Sex",
    "Cancer_Type",
    "Cancer_Stage",
    "Gene_Mutation",
    "Treatment",
    "Treatment_Response",
    "Toxicity_Risk"
]

for column in categorical_columns:

    if column in df.columns:

        print(f"\n{column}")
        print(df[column].value_counts())


# ------------------------------------------------------------
# 8. CANCER TYPE DISTRIBUTION
# ------------------------------------------------------------

print("\n[7] Cancer type distribution")

if "Cancer_Type" in df.columns:

    cancer_counts = df["Cancer_Type"].value_counts()

    plt.figure(figsize=(10, 6))

    cancer_counts.plot(kind="bar")

    plt.title("Cancer Type Distribution")

    plt.xlabel("Cancer Type")

    plt.ylabel("Number of Patients")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "cancer_type_distribution.png",
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 9. CANCER STAGE DISTRIBUTION
# ------------------------------------------------------------

print("\n[8] Cancer stage distribution")

if "Cancer_Stage" in df.columns:

    stage_counts = df["Cancer_Stage"].value_counts()

    plt.figure(figsize=(8, 5))

    stage_counts.plot(kind="bar")

    plt.title("Cancer Stage Distribution")

    plt.xlabel("Cancer Stage")

    plt.ylabel("Number of Patients")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "cancer_stage_distribution.png",
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 10. TREATMENT RESPONSE DISTRIBUTION
# ------------------------------------------------------------

print("\n[9] Treatment response distribution")

if "Treatment_Response" in df.columns:

    response_counts = df["Treatment_Response"].value_counts()

    print(response_counts)

    plt.figure(figsize=(9, 5))

    response_counts.plot(kind="bar")

    plt.title("Treatment Response Distribution")

    plt.xlabel("Treatment Response")

    plt.ylabel("Number of Patients")

    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "treatment_response_distribution.png",
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 11. TOXICITY RISK DISTRIBUTION
# ------------------------------------------------------------

print("\n[10] Toxicity risk distribution")

if "Toxicity_Risk" in df.columns:

    toxicity_counts = df["Toxicity_Risk"].value_counts()

    print(toxicity_counts)

    plt.figure(figsize=(7, 5))

    toxicity_counts.plot(kind="bar")

    plt.title("Treatment Toxicity Risk Distribution")

    plt.xlabel("Toxicity Risk")

    plt.ylabel("Number of Patients")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "toxicity_risk_distribution.png",
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 12. AGE DISTRIBUTION
# ------------------------------------------------------------

print("\n[11] Age distribution")

if "Age" in df.columns:

    plt.figure(figsize=(8, 5))

    df["Age"].plot(
        kind="hist",
        bins=20
    )

    plt.title("Patient Age Distribution")

    plt.xlabel("Age")

    plt.ylabel("Number of Patients")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "age_distribution.png",
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 13. ctDNA DISTRIBUTION
# ------------------------------------------------------------

print("\n[12] ctDNA analysis")

if "ctDNA_Level" in df.columns:

    plt.figure(figsize=(8, 5))

    df["ctDNA_Level"].plot(
        kind="hist",
        bins=25
    )

    plt.title("ctDNA Level Distribution")

    plt.xlabel("ctDNA Level")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "ctDNA_distribution.png",
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 14. BIOMARKER vs TOXICITY RISK
# ------------------------------------------------------------

print("\n[13] Biomarker vs toxicity analysis")

if "ctDNA_Level" in df.columns and "Toxicity_Risk" in df.columns:

    toxicity_order = {
        "Low": 0,
        "Moderate": 1,
        "High": 2
    }

    df["Toxicity_Numeric"] = (
        df["Toxicity_Risk"]
        .map(toxicity_order)
    )

    correlation_ctdna = df[
        ["ctDNA_Level", "Toxicity_Numeric"]
    ].corr().iloc[0, 1]

    print(
        f"ctDNA vs Toxicity correlation: "
        f"{correlation_ctdna:.4f}"
    )


# ------------------------------------------------------------
# 15. CLINICAL FEATURES vs TOXICITY
# ------------------------------------------------------------

print("\n[14] Clinical risk feature analysis")

risk_features = [
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

available_features = [
    col for col in risk_features
    if col in df.columns
]

risk_correlation = df[
    available_features + ["Toxicity_Numeric"]
].corr()["Toxicity_Numeric"].drop(
    "Toxicity_Numeric"
).sort_values(
    key=abs,
    ascending=False
)

print("\nCorrelation with Toxicity Risk:")
print(risk_correlation)


# Save risk correlation
risk_correlation.to_csv(
    EDA_DIR / "toxicity_feature_correlation.csv"
)


# ------------------------------------------------------------
# 16. CORRELATION MATRIX
# ------------------------------------------------------------

print("\n[15] Generating correlation matrix")

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 9))

plt.imshow(
    correlation_matrix,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Clinical Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    EDA_DIR / "correlation_matrix.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 17. OUTLIER DETECTION USING IQR
# ------------------------------------------------------------

print("\n[16] Outlier detection")
print("-" * 50)

outlier_results = []

for column in available_features:

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)

    upper_bound = Q3 + (1.5 * IQR)

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    outlier_count = len(outliers)

    outlier_results.append({
        "Feature": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Outlier_Count": outlier_count
    })

    print(
        f"{column}: "
        f"{outlier_count} outliers"
    )


outlier_report = pd.DataFrame(
    outlier_results
)

outlier_report.to_csv(
    EDA_DIR / "outlier_report.csv",
    index=False
)


# ------------------------------------------------------------
# 18. TOXICITY RISK BY CANCER TYPE
# ------------------------------------------------------------

print("\n[17] Toxicity risk by cancer type")

if "Cancer_Type" in df.columns:

    toxicity_by_cancer = pd.crosstab(
        df["Cancer_Type"],
        df["Toxicity_Risk"]
    )

    print(toxicity_by_cancer)

    toxicity_by_cancer.to_csv(
        EDA_DIR / "toxicity_by_cancer_type.csv"
    )


# ------------------------------------------------------------
# 19. TOXICITY RISK BY TREATMENT
# ------------------------------------------------------------

print("\n[18] Toxicity risk by treatment")

if "Treatment" in df.columns:

    toxicity_by_treatment = pd.crosstab(
        df["Treatment"],
        df["Toxicity_Risk"]
    )

    print(toxicity_by_treatment)

    toxicity_by_treatment.to_csv(
        EDA_DIR / "toxicity_by_treatment.csv"
    )


# ------------------------------------------------------------
# 20. FEATURE IMPORTANCE PREVIEW
# ------------------------------------------------------------

print("\n[19] Preliminary risk predictor ranking")
print("-" * 50)

feature_ranking = (
    risk_correlation
    .abs()
    .sort_values(ascending=False)
)

print(feature_ranking)


feature_ranking.to_csv(
    EDA_DIR / "preliminary_feature_ranking.csv"
)


# ------------------------------------------------------------
# 21. FINAL EDA SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EDA ENGINEER ANALYSIS COMPLETED")
print("=" * 70)

print("\nGenerated EDA files:")

for file in sorted(EDA_DIR.iterdir()):
    print(f"  ✓ {file.name}")

print("\nMost important preliminary risk predictors:")

print(
    feature_ranking.head(5)
)

print("\nEDA Engineer stage completed successfully.")