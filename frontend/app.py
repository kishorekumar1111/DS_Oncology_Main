import streamlit as st
import requests

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Personalized Precision Oncology",
    page_icon="🧬",
    layout="wide"
)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("🧬 Personalized Precision Oncology")
st.subheader("AI-Based Treatment Toxicity Risk Prediction")

st.write(
    "Enter the patient's clinical, genomic and treatment information "
    "to predict toxicity risk using the trained Machine Learning model."
)

# Backend API
API_URL = "http://127.0.0.1:8000/predict"

# ---------------------------------------------------------
# PATIENT INFORMATION
# ---------------------------------------------------------
st.header("👤 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=50
    )

with col2:
    sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

with col3:
    cancer_type = st.selectbox(
        "Cancer Type",
        [
            "Breast Cancer",
            "Lung Cancer",
            "Liver Cancer",
            "Colon Cancer",
            "Prostate Cancer",
            "Other"
        ]
    )

# ---------------------------------------------------------
# CANCER / GENOMIC INFORMATION
# ---------------------------------------------------------
st.header("🧬 Cancer & Genomic Information")

col1, col2, col3 = st.columns(3)

with col1:
    cancer_stage = st.selectbox(
        "Cancer Stage",
        ["I", "II", "III", "IV"]
    )

with col2:
    gene_mutation = st.selectbox(
        "Gene Mutation",
        [
            "BRCA1",
            "BRCA2",
            "EGFR",
            "KRAS",
            "TP53",
            "None"
        ]
    )

with col3:
    ctdna_level = st.number_input(
        "ctDNA Level",
        min_value=0.0,
        value=1.0,
        step=0.1
    )

# ---------------------------------------------------------
# CLINICAL DATA
# ---------------------------------------------------------
st.header("🩺 Clinical Data")

col1, col2, col3, col4 = st.columns(4)

with col1:
    protein_biomarker = st.number_input(
        "Protein Biomarker",
        min_value=0.0,
        value=30.0,
        step=0.1
    )

with col2:
    spo2 = st.number_input(
        "SpO2",
        min_value=50.0,
        max_value=100.0,
        value=97.0,
        step=0.1
    )

with col3:
    heart_rate = st.number_input(
        "Heart Rate",
        min_value=30.0,
        max_value=200.0,
        value=75.0,
        step=1.0
    )

with col4:
    wbc_count = st.number_input(
        "WBC Count",
        min_value=0.0,
        value=7.0,
        step=0.1
    )

col1, col2, col3, col4 = st.columns(4)

with col1:
    platelet_count = st.number_input(
        "Platelet Count",
        min_value=0.0,
        value=250.0,
        step=1.0
    )

with col2:
    creatinine = st.number_input(
        "Creatinine",
        min_value=0.0,
        value=1.0,
        step=0.01
    )

with col3:
    bilirubin = st.number_input(
        "Bilirubin",
        min_value=0.0,
        value=0.8,
        step=0.01
    )

with col4:
    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )

# ---------------------------------------------------------
# TREATMENT INFORMATION
# ---------------------------------------------------------
st.header("💊 Treatment Information")

col1, col2, col3 = st.columns(3)

with col1:
    ecog_status = st.number_input(
        "ECOG Status",
        min_value=0,
        max_value=5,
        value=1,
        step=1
    )

with col2:
    treatment = st.selectbox(
        "Treatment",
        [
            "Chemotherapy",
            "Immunotherapy",
            "Targeted Therapy",
            "Radiation Therapy",
            "Hormone Therapy"
        ]
    )

with col3:
    treatment_response = st.selectbox(
        "Treatment Response",
        [
            "Complete Response",
            "Partial Response",
            "Stable Disease",
            "Progressive Disease"
        ]
    )

dose_mg = st.number_input(
    "Dose (mg)",
    min_value=0.1,
    value=50.0,
    step=0.1
)

# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------
st.divider()

if st.button(
    "🔮 Predict Toxicity Risk",
    use_container_width=True
):

    patient_data = {
        "age": age,
        "sex": sex,
        "cancer_type": cancer_type,
        "cancer_stage": cancer_stage,
        "gene_mutation": gene_mutation,
        "ctDNA_level": ctdna_level,
        "protein_biomarker": protein_biomarker,
        "spo2": spo2,
        "heart_rate": heart_rate,
        "wbc_count": wbc_count,
        "platelet_count": platelet_count,
        "creatinine": creatinine,
        "bilirubin": bilirubin,
        "bmi": bmi,
        "ecog_status": ecog_status,
        "treatment": treatment,
        "dose_mg": dose_mg,
        "treatment_response": treatment_response
    }

    try:

        with st.spinner("Running AI prediction..."):

            response = requests.post(
                API_URL,
                json=patient_data,
                timeout=30
            )

        if response.status_code == 200:

            result = response.json()

            prediction = result.get("prediction")
            confidence = result.get("confidence", 0)

            st.success("Prediction generated successfully!")

            st.header("📊 Prediction Result")

            col1, col2 = st.columns(2)

            with col1:

                if prediction == "High":
                    st.error(f"🔴 {prediction} Toxicity Risk")

                elif prediction == "Moderate":
                    st.warning(f"🟠 {prediction} Toxicity Risk")

                else:
                    st.success(f"🟢 {prediction} Toxicity Risk")

            with col2:

                st.metric(
                    "Prediction Confidence",
                    f"{confidence * 100:.2f}%"
                )

            # -------------------------------------------------
            # CLASS PROBABILITIES
            # -------------------------------------------------

            probabilities = result.get(
                "class_probabilities",
                {}
            )

            if probabilities:

                st.subheader("Class Probabilities")

                for risk, probability in probabilities.items():

                    st.write(
                        f"**{risk}** — "
                        f"{probability * 100:.2f}%"
                    )

                    st.progress(
                        float(probability)
                    )

            st.info(
                "⚠️ This AI prediction is intended for research "
                "and decision-support purposes. It should not replace "
                "professional medical judgment."
            )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

            st.code(response.text)

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to the backend API. "
            "Please make sure integrationengineer.py is running."
        )

    except Exception as e:

        st.error(
            f"Unexpected error: {str(e)}"
        )