import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# PERSONALIZED PRECISION MEDICINE FOR ONCOLOGY
# STAGE 01 - EVALUATION ENGINEER
# ============================================================

print("=" * 75)
print("PERSONALIZED PRECISION ONCOLOGY - EVALUATION ENGINEER")
print("=" * 75)


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

MODEL_FILE = Path(
    "output/ml/oncology_toxicity_risk_model.pkl"
)

DATA_FILE = Path(
    "output/oncology_master_clean.csv"
)

EVAL_DIR = Path(
    "output/evaluation"
)

EVAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. LOAD MODEL
# ------------------------------------------------------------

print("\n[1] Loading trained ML model...")

pipeline = joblib.load(
    MODEL_FILE
)

print("Model loaded successfully.")


# ------------------------------------------------------------
# 3. LOAD DATA
# ------------------------------------------------------------

print("\n[2] Loading master dataset...")

df = pd.read_csv(
    DATA_FILE
)

print(
    f"Dataset shape: {df.shape}"
)


# ------------------------------------------------------------
# 4. PREPARE FEATURES AND TARGET
# ------------------------------------------------------------

print("\n[3] Preparing evaluation data...")

TARGET = "Toxicity_Risk"

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


# ------------------------------------------------------------
# 5. RECREATE UNSEEN TEST SET
# ------------------------------------------------------------

print("\n[4] Creating unseen evaluation cohort...")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)

print(
    f"Unseen evaluation samples: {len(X_test)}"
)


# ------------------------------------------------------------
# 6. GENERATE PREDICTIONS
# ------------------------------------------------------------

print("\n[5] Generating unseen patient predictions...")

y_pred = pipeline.predict(
    X_test
)

y_probability = pipeline.predict_proba(
    X_test
)

classes = pipeline.classes_


# ------------------------------------------------------------
# 7. BASIC PERFORMANCE
# ------------------------------------------------------------

print("\n[6] Basic performance evaluation...")

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# 8. CONFUSION MATRIX
# ------------------------------------------------------------

print("\n[7] Creating confusion matrix...")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)

print("\nConfusion Matrix:")
print(cm)


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
    "Evaluation Confusion Matrix"
)

plt.xlabel(
    "Predicted Risk"
)

plt.ylabel(
    "Actual Risk"
)

plt.tight_layout()

plt.savefig(
    EVAL_DIR / "evaluation_confusion_matrix.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 9. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n[8] Class-wise evaluation...")

classification = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(
    classification
)

with open(
    EVAL_DIR / "evaluation_classification_report.txt",
    "w"
) as file:

    file.write(
        classification
    )


# ------------------------------------------------------------
# 10. MCC
# ------------------------------------------------------------

print(
    f"\nMatthews Correlation Coefficient: {mcc:.4f}"
)


# ------------------------------------------------------------
# 11. CROSS VALIDATION
# ------------------------------------------------------------

print("\n[9] Running cross-validation stress test...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1
)

print(
    "\n5-Fold Cross Validation F1 Scores:"
)

for i, score in enumerate(
    cv_scores,
    start=1
):

    print(
        f"Fold {i}: {score:.4f}"
    )


cv_mean = cv_scores.mean()

cv_std = cv_scores.std()

print(
    f"\nCV Mean F1 : {cv_mean:.4f}"
)

print(
    f"CV Std     : {cv_std:.4f}"
)


# ------------------------------------------------------------
# 12. CONFIDENCE ANALYSIS
# ------------------------------------------------------------

print("\n[10] Checking model confidence...")

max_confidence = np.max(
    y_probability,
    axis=1
)

prediction_confidence = (
    max_confidence
)


# ------------------------------------------------------------
# 13. CORRECT / WRONG PREDICTIONS
# ------------------------------------------------------------

correct_prediction = (
    y_pred == y_test.values
)


# ------------------------------------------------------------
# 14. OVERCONFIDENT WRONG PREDICTIONS
# ------------------------------------------------------------

print(
    "\n[11] Detecting overconfident wrong predictions..."
)

# Prediction is considered highly confident
# when probability >= 0.80

HIGH_CONFIDENCE_THRESHOLD = 0.80

overconfident_mask = (

    (max_confidence >= HIGH_CONFIDENCE_THRESHOLD)

    &

    (~correct_prediction)
)


overconfident_count = (
    overconfident_mask.sum()
)


print(
    f"High-confidence wrong predictions: "
    f"{overconfident_count}"
)


# ------------------------------------------------------------
# 15. CONFIDENCE GROUPS
# ------------------------------------------------------------

confidence_groups = []

for confidence in max_confidence:

    if confidence < 0.50:

        group = "Low Confidence"

    elif confidence < 0.80:

        group = "Moderate Confidence"

    else:

        group = "High Confidence"

    confidence_groups.append(
        group
    )


# ------------------------------------------------------------
# 16. ERROR ANALYSIS DATAFRAME
# ------------------------------------------------------------

print("\n[12] Creating error analysis report...")

error_analysis = X_test.copy()

error_analysis["Actual_Risk"] = (
    y_test.values
)

error_analysis["Predicted_Risk"] = (
    y_pred
)

error_analysis["Prediction_Confidence"] = (
    max_confidence
)

error_analysis["Confidence_Level"] = (
    confidence_groups
)

error_analysis["Correct_Prediction"] = (
    correct_prediction
)

error_analysis["Overconfident_Error"] = (
    overconfident_mask
)


# ------------------------------------------------------------
# 17. SAVE ALL ERRORS
# ------------------------------------------------------------

all_errors = error_analysis[
    error_analysis["Correct_Prediction"] == False
]

all_errors.to_csv(
    EVAL_DIR / "all_prediction_errors.csv",
    index=False
)

print(
    f"Total prediction errors: "
    f"{len(all_errors)}"
)


# ------------------------------------------------------------
# 18. SAVE OVERCONFIDENT ERRORS
# ------------------------------------------------------------

overconfident_errors = error_analysis[
    error_analysis["Overconfident_Error"] == True
]

overconfident_errors.to_csv(
    EVAL_DIR / "overconfident_errors.csv",
    index=False
)

print(
    f"Overconfident errors saved: "
    f"{len(overconfident_errors)}"
)


# ------------------------------------------------------------
# 19. CONFIDENCE ANALYSIS
# ------------------------------------------------------------

print("\n[13] Confidence statistics...")

confidence_statistics = pd.DataFrame({

    "Metric": [
        "Average Confidence",
        "Minimum Confidence",
        "Maximum Confidence",
        "High Confidence Predictions",
        "Moderate Confidence Predictions",
        "Low Confidence Predictions",
        "High Confidence Wrong Predictions"
    ],

    "Value": [

        max_confidence.mean(),

        max_confidence.min(),

        max_confidence.max(),

        np.sum(
            max_confidence >= 0.80
        ),

        np.sum(
            (max_confidence >= 0.50)
            &
            (max_confidence < 0.80)
        ),

        np.sum(
            max_confidence < 0.50
        ),

        overconfident_count
    ]
})


print(
    confidence_statistics
)


confidence_statistics.to_csv(
    EVAL_DIR / "confidence_analysis.csv",
    index=False
)


# ------------------------------------------------------------
# 20. CONFIDENCE VISUALIZATION
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    max_confidence,
    bins=20
)

plt.title(
    "Model Prediction Confidence Distribution"
)

plt.xlabel(
    "Prediction Confidence"
)

plt.ylabel(
    "Number of Patients"
)

plt.tight_layout()

plt.savefig(
    EVAL_DIR / "confidence_distribution.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 21. ERROR RATE BY CONFIDENCE
# ------------------------------------------------------------

print(
    "\n[14] Confidence vs error analysis..."
)

confidence_bins = [
    0.0,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.01
]

confidence_labels = [
    "<50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%"
]

confidence_df = pd.DataFrame({

    "Confidence": max_confidence,

    "Correct": correct_prediction
})


confidence_df["Confidence_Range"] = pd.cut(

    confidence_df["Confidence"],

    bins=confidence_bins,

    labels=confidence_labels,

    right=False
)


confidence_summary = (
    confidence_df
    .groupby(
        "Confidence_Range",
        observed=False
    )
    .agg(
        Total_Predictions=("Correct", "count"),
        Correct=("Correct", "sum")
    )
)


confidence_summary[
    "Errors"
] = (
    confidence_summary["Total_Predictions"]
    -
    confidence_summary["Correct"]
)


confidence_summary[
    "Error_Rate"
] = (

    confidence_summary["Errors"]
    /
    confidence_summary["Total_Predictions"]
)


print(
    confidence_summary
)


confidence_summary.to_csv(
    EVAL_DIR / "confidence_error_rate.csv"
)


# ------------------------------------------------------------
# 22. CLASS-WISE PERFORMANCE
# ------------------------------------------------------------

print(
    "\n[15] Class-wise performance analysis..."
)

class_precision = precision_score(
    y_test,
    y_pred,
    labels=classes,
    average=None,
    zero_division=0
)

class_recall = recall_score(
    y_test,
    y_pred,
    labels=classes,
    average=None,
    zero_division=0
)

class_f1 = f1_score(
    y_test,
    y_pred,
    labels=classes,
    average=None,
    zero_division=0
)


class_performance = pd.DataFrame({

    "Class": classes,

    "Precision": class_precision,

    "Recall": class_recall,

    "F1_Score": class_f1
})


print(
    class_performance
)


class_performance.to_csv(
    EVAL_DIR / "class_wise_performance.csv",
    index=False
)


# ------------------------------------------------------------
# 23. EVALUATION METRICS
# ------------------------------------------------------------

evaluation_metrics = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Macro Precision",

        "Macro Recall",

        "Macro F1",

        "MCC",

        "Cross Validation Mean F1",

        "Cross Validation Std",

        "Average Prediction Confidence",

        "Overconfident Wrong Predictions"
    ],

    "Score": [

        accuracy,

        precision,

        recall,

        f1,

        mcc,

        cv_mean,

        cv_std,

        max_confidence.mean(),

        overconfident_count
    ]
})


evaluation_metrics.to_csv(
    EVAL_DIR / "evaluation_metrics.csv",
    index=False
)


# ------------------------------------------------------------
# 24. OVERCONFIDENCE FLAG
# ------------------------------------------------------------

print(
    "\n[16] Evaluating model overconfidence..."
)

if overconfident_count == 0:

    overconfidence_status = (
        "No high-confidence wrong predictions detected"
    )

elif overconfident_count <= 5:

    overconfidence_status = (
        "Low number of high-confidence errors"
    )

elif overconfident_count <= 15:

    overconfidence_status = (
        "Moderate overconfidence detected"
    )

else:

    overconfidence_status = (
        "High overconfidence detected - review required"
    )


print(
    f"Overconfidence status: "
    f"{overconfidence_status}"
)


# ------------------------------------------------------------
# 25. FINAL EVALUATION REPORT
# ------------------------------------------------------------

report_lines = [

    "PERSONALIZED PRECISION ONCOLOGY",

    "STAGE 01 - EVALUATION ENGINEER",

    "",

    "MODEL EVALUATION REPORT",

    "=======================",

    "",

    f"Accuracy: {accuracy:.4f}",

    f"Macro Precision: {precision:.4f}",

    f"Macro Recall: {recall:.4f}",

    f"Macro F1: {f1:.4f}",

    f"MCC: {mcc:.4f}",

    "",

    f"Cross Validation Mean F1: {cv_mean:.4f}",

    f"Cross Validation Std: {cv_std:.4f}",

    "",

    f"Average Prediction Confidence: "
    f"{max_confidence.mean():.4f}",

    f"High Confidence Wrong Predictions: "
    f"{overconfident_count}",

    "",

    f"Overconfidence Status: "
    f"{overconfidence_status}",

    "",

    "Evaluation completed on an unseen test cohort.",

    "Errors and high-confidence errors were saved "
    "for further investigation."
]


with open(
    EVAL_DIR / "evaluation_report.txt",
    "w"
) as file:

    file.write(
        "\n".join(report_lines)
    )


# ------------------------------------------------------------
# 26. FINAL OUTPUT
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("EVALUATION ENGINEER STAGE COMPLETED")
print("=" * 75)

print(
    f"\nAccuracy              : {accuracy:.4f}"
)

print(
    f"Macro Precision       : {precision:.4f}"
)

print(
    f"Macro Recall          : {recall:.4f}"
)

print(
    f"Macro F1              : {f1:.4f}"
)

print(
    f"MCC                   : {mcc:.4f}"
)

print(
    f"CV Mean F1            : {cv_mean:.4f}"
)

print(
    f"CV Std                : {cv_std:.4f}"
)

print(
    f"Overconfident Errors  : {overconfident_count}"
)

print(
    f"\nStatus: {overconfidence_status}"
)


print("\nGenerated evaluation files:")

for file in sorted(
    EVAL_DIR.iterdir()
):

    print(
        f"  ✓ {file.name}"
    )


print(
    "\nEvaluation Engineer stage "
    "completed successfully."
)
