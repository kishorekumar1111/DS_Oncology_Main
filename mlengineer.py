import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    classification_report
)


# ============================================================
# PERSONALIZED PRECISION MEDICINE FOR ONCOLOGY
# STAGE 01 - ML ENGINEER
# ============================================================

print("=" * 80)
print("PERSONALIZED PRECISION ONCOLOGY - IMPROVED ML ENGINEER")
print("=" * 80)


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = Path("output/oncology_master_clean.csv")

ML_DIR = Path("output/ml")
ML_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n[1] Loading clean master dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 3. BASIC DATA CHECK
# ============================================================

print("\n[2] Checking dataset...")

print("\nMissing values:")
print(df.isnull().sum().sum())

print(
    f"Total duplicate rows: {df.duplicated().sum()}"
)


# ============================================================
# 4. TARGET
# ============================================================

TARGET = "Toxicity_Risk"

print("\n[3] Target variable:")
print(TARGET)

print("\nTarget distribution:")
print(df[TARGET].value_counts())


# ============================================================
# 5. REMOVE NON-PREDICTIVE COLUMNS
# ============================================================

print("\n[4] Preparing features...")

columns_to_remove = [
    "Patient_ID",
    "Record_Timestamp",
    TARGET
]

X = df.drop(
    columns=[
        col for col in columns_to_remove
        if col in df.columns
    ]
)

y = df[TARGET]


# ============================================================
# 6. TARGET CLASSES
# ============================================================

print("\n[5] Target classes:")
print(y.unique())


# ============================================================
# 7. DATA TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

print("\n[6] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# 9. PREPROCESSING
# ============================================================

print("\n[7] Creating preprocessing pipeline...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 10. MODEL 1 - RANDOM FOREST
# ============================================================

print("\n[8] Creating optimized Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=800,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)


rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "classifier",
            rf_model
        )
    ]
)


# ============================================================
# 11. MODEL 2 - EXTRA TREES
# ============================================================

print("\n[9] Creating Extra Trees model...")

et_model = ExtraTreesClassifier(
    n_estimators=800,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


et_pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "classifier",
            et_model
        )
    ]
)


# ============================================================
# 12. TRAIN RANDOM FOREST
# ============================================================

print("\n[10] Training Random Forest...")

rf_pipeline.fit(
    X_train,
    y_train
)

rf_pred = rf_pipeline.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print(
    f"Random Forest Accuracy: {rf_accuracy:.4f}"
)


# ============================================================
# 13. TRAIN EXTRA TREES
# ============================================================

print("\n[11] Training Extra Trees...")

et_pipeline.fit(
    X_train,
    y_train
)

et_pred = et_pipeline.predict(X_test)

et_accuracy = accuracy_score(
    y_test,
    et_pred
)

print(
    f"Extra Trees Accuracy: {et_accuracy:.4f}"
)


# ============================================================
# 14. SELECT BEST MODEL
# ============================================================

print("\n[12] Selecting best model...")

if et_accuracy > rf_accuracy:

    pipeline = et_pipeline
    y_pred = et_pred

    print("✓ Best model: Extra Trees")

else:

    pipeline = rf_pipeline
    y_pred = rf_pred

    print("✓ Best model: Random Forest")


# ============================================================
# 15. PROBABILITIES
# ============================================================

print("\n[13] Generating prediction probabilities...")

y_probability = pipeline.predict_proba(
    X_test
)


# ============================================================
# 16. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# 17. RECALL
# ============================================================

recall_macro = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# 18. F1
# ============================================================

f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# 19. MCC
# ============================================================

mcc = matthews_corrcoef(
    y_test,
    y_pred
)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

classes = pipeline.classes_

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)


# ============================================================
# 21. SPECIFICITY
# ============================================================

specificities = []

for i in range(len(classes)):

    TP = cm[i, i]

    FN = cm[i, :].sum() - TP

    FP = cm[:, i].sum() - TP

    TN = cm.sum() - (
        TP + FN + FP
    )

    specificity = (
        TN / (TN + FP)
        if (TN + FP) > 0
        else 0
    )

    specificities.append(
        specificity
    )


specificity_macro = np.mean(
    specificities
)


# ============================================================
# 22. ROC-AUC
# ============================================================

try:

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
        multi_class="ovr",
        average="macro"
    )

except ValueError:

    roc_auc = np.nan


# ============================================================
# 23. PR-AUC
# ============================================================

try:

    y_test_encoded = pd.get_dummies(
        y_test
    ).reindex(
        columns=classes,
        fill_value=0
    )

    pr_auc = average_precision_score(
        y_test_encoded,
        y_probability,
        average="macro"
    )

except ValueError:

    pr_auc = np.nan


# ============================================================
# 24. MODEL PERFORMANCE
# ============================================================

print("\n")
print("=" * 80)
print("FINAL MODEL PERFORMANCE")
print("=" * 80)

print(
    f"\nAccuracy       : {accuracy:.4f}"
)

print(
    f"ROC-AUC        : {roc_auc:.4f}"
)

print(
    f"PR-AUC         : {pr_auc:.4f}"
)

print(
    f"Recall         : {recall_macro:.4f}"
)

print(
    f"Specificity    : {specificity_macro:.4f}"
)

print(
    f"F1 Score       : {f1_macro:.4f}"
)

print(
    f"MCC            : {mcc:.4f}"
)


# ============================================================
# 25. CLASSIFICATION REPORT
# ============================================================

print("\n[14] Classification report:")

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)


with open(
    ML_DIR / "classification_report.txt",
    "w"
) as file:

    file.write(report)


# ============================================================
# 26. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "ROC-AUC",
        "PR-AUC",
        "Recall",
        "Specificity",
        "F1 Score",
        "MCC"
    ],

    "Score": [
        accuracy,
        roc_auc,
        pr_auc,
        recall_macro,
        specificity_macro,
        f1_macro,
        mcc
    ]

})


metrics.to_csv(
    ML_DIR / "model_metrics.csv",
    index=False
)


# ============================================================
# 27. CONFUSION MATRIX
# ============================================================

print("\n[15] Creating confusion matrix...")

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes
)

plt.title(
    "Toxicity Risk Confusion Matrix"
)

plt.xlabel(
    "Predicted Risk"
)

plt.ylabel(
    "Actual Risk"
)

plt.tight_layout()

plt.savefig(
    ML_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 28. FEATURE IMPORTANCE
# ============================================================

print("\n[16] Calculating feature importance...")

trained_model = pipeline.named_steps[
    "classifier"
]

trained_preprocessor = pipeline.named_steps[
    "preprocessing"
]

feature_names = (
    trained_preprocessor
    .get_feature_names_out()
)

importance_values = (
    trained_model.feature_importances_
)

feature_importance = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance_values

})

feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\nTop 20 important features:")

print(
    feature_importance.head(20)
)


feature_importance.to_csv(
    ML_DIR / "feature_importance.csv",
    index=False
)


# ============================================================
# 29. FEATURE IMPORTANCE GRAPH
# ============================================================

print("\n[17] Creating feature importance graph...")

top_features = (
    feature_importance
    .head(15)
    .sort_values(
        "Importance"
    )
)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title(
    "Top Clinical & Genomic Risk Predictors"
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.savefig(
    ML_DIR / "feature_importance.png",
    dpi=300
)

plt.close()


# ============================================================
# 30. TEST PREDICTIONS
# ============================================================

print("\n[18] Saving patient risk predictions...")

prediction_output = X_test.copy()

prediction_output[
    "Actual_Toxicity_Risk"
] = y_test.values

prediction_output[
    "Predicted_Toxicity_Risk"
] = y_pred


prediction_output.to_csv(
    ML_DIR / "test_predictions.csv",
    index=False
)


# ============================================================
# 31. SAVE MODEL
# ============================================================

print("\n[19] Saving trained ML model...")

MODEL_FILE = (
    ML_DIR /
    "oncology_toxicity_risk_model.pkl"
)

joblib.dump(
    pipeline,
    MODEL_FILE
)

print(
    f"✓ Model saved to: {MODEL_FILE}"
)


# ============================================================
# 32. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("IMPROVED ML ENGINEER STAGE COMPLETED")
print("=" * 80)

print(
    f"\nBest Model     : {type(trained_model).__name__}"
)

print(
    f"Target         : {TARGET}"
)

print(
    "Classes        : Low / Moderate / High"
)

print(
    f"\nAccuracy       : {accuracy:.4f}"
)

print(
    f"ROC-AUC        : {roc_auc:.4f}"
)

print(
    f"PR-AUC         : {pr_auc:.4f}"
)

print(
    f"Recall         : {recall_macro:.4f}"
)

print(
    f"Specificity    : {specificity_macro:.4f}"
)

print(
    f"F1 Score       : {f1_macro:.4f}"
)

print(
    f"MCC            : {mcc:.4f}"
)


print("\nGenerated files:")

for file in sorted(ML_DIR.iterdir()):

    print(
        f"  ✓ {file.name}"
    )


print(
    "\nML Engineer stage completed successfully."
)