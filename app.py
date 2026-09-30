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


@st.cache_resource
def load_models():
    try:
        model = joblib.load(
            MODELS_DIR / "final_random_forest.pkl"
        )

        preprocessor = joblib.load(
            MODELS_DIR / "final_preprocessor.pkl"
        )

        calibrator = joblib.load(
            MODELS_DIR / "final_calibrator.pkl"
        )

        return model, preprocessor, calibrator

    except Exception as e:
        st.error(
            "Unable to load the model files. "
            "Please check that all model artifacts are present."
        )
        st.stop()


model, preprocessor, calibrator = load_models()


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
# Get Categories Learned by Model
# --------------------------------------------------

cat_encoder = (
    preprocessor
    .named_transformers_["cat"]
    .named_steps["encoder"]
)

feature_categories = cat_encoder.categories_

borr_state_options = sorted(
    feature_categories[0].tolist()
)

processing_method_options = sorted(
    feature_categories[1].tolist()
)

project_state_options = sorted(
    feature_categories[3].tolist()
)

district_office_options = sorted(
    feature_categories[4].tolist()
)

business_type_options = sorted(
    feature_categories[5].tolist()
)

business_age_options = sorted(
    feature_categories[6].tolist()
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

    borr_state = st.selectbox(
        "Borrower State",
        borr_state_options,
        index=(
            borr_state_options.index("CA")
            if "CA" in borr_state_options
            else 0
        )
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

    processing_method = st.selectbox(
        "Processing Method",
        processing_method_options,
        index=(
            processing_method_options.index("7a General")
            if "7a General" in processing_method_options
            else 0
        )
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

    business_type = st.selectbox(
        "Business Type",
        business_type_options,
        index=(
            business_type_options.index("CORPORATION")
            if "CORPORATION" in business_type_options
            else 0
        )
    )

    business_age = st.selectbox(
        "Business Age",
        business_age_options,
        index=(
            business_age_options.index(
                "Existing, 5 or more years"
            )
            if "Existing, 5 or more years"
            in business_age_options
            else 0
        )
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

    project_state = st.selectbox(
        "Project State",
        project_state_options,
        index=(
            project_state_options.index("CA")
            if "CA" in project_state_options
            else 0
        )
    )

    district_office = st.selectbox(
        "SBA District Office",
        district_office_options
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

        st.error(
            "Please select a Borrower State."
        )

        st.stop()


    if not project_state.strip():

        st.error(
            "Please select a Project State."
        )

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

        "BorrState":
            borr_state.strip().upper(),

        "GrossApproval":
            gross_approval,

        "SBAGuaranteedApproval":
            sba_guaranteed,

        "TermInMonths":
            term,

        "JobsSupported":
            jobs_supported,

        "ApprovalMonth":
            approval_month,

        "ApprovalYear":
            approval_year,

        "ProcessingMethod":
            processing_method,

        "NaicsSector":
            naics_sector,

        "ProjectState":
            project_state.strip().upper(),

        "SBADistrictOffice":
            district_office,

        "BusinessType":
            business_type,

        "BusinessAge":
            business_age,

        "RevolverStatus":
            revolver_status,

        "CollateralInd":
            collateral,

        "TermGroup":
            term_group
    }])


    # ----------------------------------------------
    # Preprocess and Predict
    # ----------------------------------------------

    try:

        processed_input = preprocessor.transform(
            input_data
        )

        raw_probability = model.predict_proba(
            processed_input
        )[0][1]

        probability = calibrator.predict(
            [raw_probability]
        )[0]

    except Exception:

        st.error(
            "An error occurred while making the prediction. "
            "Please check the entered values and try again."
        )

        st.stop()


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
    