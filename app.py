import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SBA Loan Default Predictor",
    page_icon="🏦",
    layout="wide"
)


# --------------------------------------------------
# Load Final Model Artifacts
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

model = joblib.load(
    MODELS_DIR / "final_random_forest.pkl"
)

preprocessor = joblib.load(
    MODELS_DIR / "final_preprocessor.pkl"
)

calibrator = joblib.load(
    MODELS_DIR / "final_calibrator.pkl"
)


# --------------------------------------------------
# NAICS Sector Mapping
# --------------------------------------------------

NAICS_SECTOR_MAP = {
    "11": "Agriculture, Forestry, Fishing",
    "21": "Mining, Oil & Gas",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "32": "Manufacturing",
    "33": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade",
    "45": "Retail Trade",
    "48": "Transportation & Warehousing",
    "49": "Transportation & Warehousing",
    "51": "Information",
    "52": "Finance & Insurance",
    "53": "Real Estate",
    "54": "Professional Services",
    "55": "Management of Companies",
    "56": "Administrative & Support Services",
    "61": "Educational Services",
    "62": "Health Care & Social Assistance",
    "71": "Arts & Entertainment",
    "72": "Accommodation & Food Services",
    "81": "Other Services",
    "92": "Public Administration"
}


# --------------------------------------------------
# Create Term Group
# --------------------------------------------------

def get_term_group(term):

    if term <= 24:
        return "0-24"
    elif term <= 60:
        return "25-60"
    elif term <= 84:
        return "61-84"
    elif term <= 120:
        return "85-120"
    elif term <= 180:
        return "121-180"
    elif term <= 240:
        return "181-240"
    elif term <= 300:
        return "241-300"
    else:
        return ">300"


# --------------------------------------------------
# Convert NAICS Code to Sector
# --------------------------------------------------

def get_naics_sector(naics_code):

    naics_code = str(naics_code).strip()

    if len(naics_code) < 2:
        return "Unknown"

    sector_code = naics_code[:2]

    return NAICS_SECTOR_MAP.get(
        sector_code,
        "Unknown"
    )


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏦 SBA Loan Default Predictor")

st.write(
    "Enter the loan and business details to estimate "
    "the probability of loan default."
)

st.caption(
    "The model uses information available at loan approval "
    "and was evaluated using a time-based split."
)


# --------------------------------------------------
# Loan Information
# --------------------------------------------------

st.header("Loan Information")

col1, col2, col3 = st.columns(3)


with col1:

    borr_state = st.text_input(
        "Borrower State",
        value="CA"
    )

    gross_approval = st.number_input(
        "Gross Approval Amount",
        min_value=0.0,
        value=150000.0,
        step=1000.0
    )

    sba_guaranteed = st.number_input(
        "SBA Guaranteed Approval",
        min_value=0.0,
        value=112500.0,
        step=1000.0
    )


with col2:

    processing_method = st.text_input(
        "Processing Method",
        value="7a General"
    )

    term = st.number_input(
        "Term in Months",
        min_value=0.0,
        value=60.0,
        step=1.0
    )

    jobs_supported = st.number_input(
        "Jobs Supported",
        min_value=0.0,
        value=10.0,
        step=1.0
    )


with col3:

    approval_month = st.number_input(
        "Approval Month",
        min_value=1,
        max_value=12,
        value=6,
        step=1
    )

    approval_year = st.number_input(
        "Approval Year",
        min_value=1999,
        max_value=2009,
        value=2007,
        step=1
    )


# --------------------------------------------------
# Business Information
# --------------------------------------------------

st.header("Business Information")

col1, col2, col3 = st.columns(3)


with col1:

    business_type = st.text_input(
        "Business Type",
        value="CORPORATION"
    )

    business_age = st.text_input(
        "Business Age",
        value="Existing, 5 or more years"
    )

    revolver_status = st.selectbox(
        "Revolver Status",
        ["N", "Y"]
    )


with col2:

    collateral = st.selectbox(
        "Collateral",
        ["N", "Y"]
    )

    project_state = st.text_input(
        "Project State",
        value="CA"
    )

    district_office = st.text_input(
        "SBA District Office",
        value="SOUTH FLORIDA DISTRICT OFFICE"
    )


with col3:

    naics_code = st.text_input(
        "NAICS Code",
        value="722511"
    )

    naics_sector = get_naics_sector(
        naics_code
    )

    st.info(
        f"Detected NAICS Sector: {naics_sector}"
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "🔍 Predict Default Risk",
    use_container_width=True
):

    # ----------------------------------------------
    # Basic Validation
    # ----------------------------------------------

    if sba_guaranteed > gross_approval:

        st.error(
            "SBA Guaranteed Approval cannot be greater "
            "than Gross Approval."
        )

        st.stop()


    if not borr_state.strip():

        st.error("Please enter a Borrower State.")

        st.stop()


    if not project_state.strip():

        st.error("Please enter a Project State.")

        st.stop()


    # ----------------------------------------------
    # Feature Engineering
    # ----------------------------------------------

    term_group = get_term_group(term)

    naics_sector = get_naics_sector(
        naics_code
    )


    # ----------------------------------------------
    # Create Input DataFrame
    # ----------------------------------------------

    input_data = pd.DataFrame([{

        "BorrState": borr_state.strip().upper(),

        "GrossApproval": gross_approval,

        "SBAGuaranteedApproval": sba_guaranteed,

        "TermInMonths": term,

        "JobsSupported": jobs_supported,

        "ApprovalMonth": approval_month,

        "ApprovalYear": approval_year,

        "ProcessingMethod": processing_method,

        "NaicsSector": naics_sector,

        "ProjectState": project_state.strip().upper(),

        "SBADistrictOffice": district_office,

        "BusinessType": business_type,

        "BusinessAge": business_age,

        "RevolverStatus": revolver_status,

        "CollateralInd": collateral,

        "TermGroup": term_group

    }])


    # ----------------------------------------------
    # Preprocess
    # ----------------------------------------------

    processed_input = preprocessor.transform(
        input_data
    )


    # ----------------------------------------------
    # Raw Random Forest Probability
    # ----------------------------------------------

    raw_probability = model.predict_proba(
        processed_input
    )[0][1]


    # ----------------------------------------------
    # Calibrated Probability
    # ----------------------------------------------

    probability = calibrator.predict(
        [raw_probability]
    )[0]


    # ----------------------------------------------
    # Final Prediction
    # ----------------------------------------------

    prediction = int(
        probability >= 0.50
    )


    # ----------------------------------------------
    # Display Result
    # ----------------------------------------------

    st.divider()

    st.header("Prediction Result")

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Estimated Default Probability",
            f"{probability * 100:.2f}%"
        )


    with col2:

        if prediction == 1:

            st.error(
                "⚠️ Predicted: DEFAULT"
            )

        else:

            st.success(
                "✅ Predicted: NO DEFAULT"
            )


    st.progress(
        float(probability)
    )

    st.caption(
        "Classification threshold: 50%. "
        "The displayed probability is calibrated using "
        "a separate historical calibration period."
    )