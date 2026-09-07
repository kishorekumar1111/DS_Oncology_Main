from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime


# ============================================================
# PERSONALIZED PRECISION MEDICINE FOR ONCOLOGY
# STAGE 01 - INTEGRATION ENGINEER
# ============================================================


# ------------------------------------------------------------
# 1. APPLICATION CONFIGURATION
# ------------------------------------------------------------

MODEL_FILE = Path(
    "output/ml/oncology_toxicity_risk_model.pkl"
)


# ------------------------------------------------------------
# 2. LOAD TRAINED ML MODEL
# ------------------------------------------------------------

print("=" * 70)
print("PERSONALIZED PRECISION ONCOLOGY")
print("STAGE 01 - INTEGRATION ENGINEER")
print("=" * 70)

print("\nLoading trained ML model...")

try:

    model = joblib.load(
        MODEL_FILE
    )

    print("✓ ML model loaded successfully.")

except Exception as error:

    print(
        f"✗ Model loading failed: {error}"
    )

    model = None


# ------------------------------------------------------------
# 3. CREATE FASTAPI APPLICATION
# ------------------------------------------------------------

app = FastAPI(

    title="Personalized Precision Oncology API",

    description=(
        "Real-time oncology treatment toxicity "
        "risk prediction API."
    ),

    version="1.0.0"
)


# ------------------------------------------------------------
# 4. INPUT DATA MODEL
# ------------------------------------------------------------

class PatientData(BaseModel):

    age: float = Field(
        ...,
        ge=18,
        le=100,
        description="Patient age"
    )

    sex: str

    cancer_type: str

    cancer_stage: str

    gene_mutation: str

    ctDNA_level: float = Field(
        ...,
        ge=0,
        description="ctDNA biomarker level"
    )

    protein_biomarker: float

    spo2: float = Field(
        ...,
        ge=50,
        le=100,
        description="Blood oxygen saturation"
    )

    heart_rate: float = Field(
        ...,
        ge=30,
        le=200
    )

    wbc_count: float = Field(
        ...,
        ge=0
    )

    platelet_count: float = Field(
        ...,
        ge=0
    )

    creatinine: float = Field(
        ...,
        ge=0
    )

    bilirubin: float = Field(
        ...,
        ge=0
    )

    bmi: float = Field(
        ...,
        ge=10,
        le=60
    )

    ecog_status: int = Field(
        ...,
        ge=0,
        le=5
    )

    treatment: str

    treatment_response: str

    dose_mg: float = Field(
        ...,
        gt=0
    )


# ------------------------------------------------------------
# 5. ROOT ENDPOINT
# ------------------------------------------------------------

@app.get("/")
def home():

    return {

        "system":
        "Personalized Precision Oncology",

        "stage":
        "Stage 01 - Integration Engineer",

        "status":
        "API is running",

        "model":
        "Random Forest Toxicity Risk Classifier",

        "prediction_classes":
        [
            "Low",
            "Moderate",
            "High"
        ],

        "timestamp":
        datetime.utcnow().isoformat()
    }


# ------------------------------------------------------------
# 6. HEALTH CHECK ENDPOINT
# ------------------------------------------------------------

@app.get("/health")
def health_check():

    if model is None:

        return {

            "status": "error",

            "model_loaded": False

        }

    return {

        "status": "healthy",

        "model_loaded": True,

        "service":
        "Oncology Toxicity Prediction API",

        "timestamp":
        datetime.utcnow().isoformat()

    }


# ------------------------------------------------------------
# 7. PREDICTION ENDPOINT
# ------------------------------------------------------------

@app.post("/predict")
def predict_toxicity(
    patient: PatientData
):

    # Check model
    if model is None:

        raise HTTPException(

            status_code=500,

            detail="ML model is not loaded."

        )


    # --------------------------------------------------------
    # Convert incoming patient data
    # into model-compatible format
    # --------------------------------------------------------

    patient_data = {

        "Age":
        patient.age,

        "Sex":
        patient.sex,

        "Cancer_Type":
        patient.cancer_type,

        "Cancer_Stage":
        patient.cancer_stage,

        "Gene_Mutation":
        patient.gene_mutation,

        "ctDNA_Level":
        patient.ctDNA_level,

        "Protein_Biomarker":
        patient.protein_biomarker,

        "SpO2":
        patient.spo2,

        "Heart_Rate":
        patient.heart_rate,

        "WBC_Count":
        patient.wbc_count,

        "Platelet_Count":
        patient.platelet_count,

        "Creatinine":
        patient.creatinine,

        "Bilirubin":
        patient.bilirubin,

        "BMI":
        patient.bmi,

        "ECOG_Status":
        patient.ecog_status,

        "Treatment":
        patient.treatment,

        "Treatment_Response":
        patient.treatment_response,

        "Dose_mg":
        patient.dose_mg
    }


    input_df = pd.DataFrame(
        [patient_data]
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            input_df
        )[0]

        probabilities = model.predict_proba(
            input_df
        )[0]

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=f"Prediction failed: {error}"

        )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = float(
        np.max(probabilities)
    )


    # --------------------------------------------------------
    # CLASS PROBABILITIES
    # --------------------------------------------------------

    classes = model.classes_

    class_probabilities = {

        str(class_name):
        round(
            float(probability),
            4
        )

        for class_name, probability
        in zip(
            classes,
            probabilities
        )
    }


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "status":
        "success",

        "prediction":
        str(prediction),

        "confidence":
        round(
            confidence,
            4
        ),

        "class_probabilities":
        class_probabilities,

        "timestamp":
        datetime.utcnow().isoformat(),

        "message":
        "Prediction generated successfully."
    }


# ------------------------------------------------------------
# 8. API INFORMATION
# ------------------------------------------------------------

@app.get("/model-info")
def model_info():

    return {

        "model_type":
        "Random Forest Classifier",

        "purpose":
        "Treatment Toxicity Risk Classification",

        "classes":
        [
            "Low",
            "Moderate",
            "High"
        ],

        "input_type":
        "Clinical + Genomic + Treatment Data",

        "output_type":
        "Toxicity Risk",

        "real_time":
        True
    }


# ------------------------------------------------------------
# 9. STARTUP MESSAGE
# ------------------------------------------------------------

print("\nAPI configuration completed.")

print(
    "\nAvailable endpoints:"
)

print(
    "GET  /"
)

print(
    "GET  /health"
)

print(
    "GET  /model-info"
)

print(
    "POST /predict"
)

print(
    "\nIntegration Engineer module ready."
)