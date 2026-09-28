import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Load Model and Preprocessor
# --------------------------------------------------

model = joblib.load("models/random_forest_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


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
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SBA Loan Default Predictor",
    page_icon="🏦",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏦 SBA Loan Default Predictor")

st.write(
    "Enter the loan and business details to predict "
    "the probability of loan default."
)


# --------------------------------------------------
# Loan Information
# --------------------------------------------------

st.header("Loan Information")

col1, col2, col3 = st.columns(3)


with col1:

    borr_state = st.selectbox(
        "Borrower State",
        [
            "AL", "CA", "FL", "GA", "IL", "NY",
            "NC", "OH", "PA", "TX", "VA", "WA", "Other"
        ]
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

    approval_fy = st.number_input(
        "Approval FY",
        min_value=1999,
        max_value=2030,
        value=2007,
        step=1
    )


with col2:

    processing_method = st.selectbox(
        "Processing Method",
        [
            "SBA Express Program",
            "Preferred Lenders Program",
            "7a General",
            "Community Express",
            "Low Documentation Program",
            "Certified Lenders Program",
            "International Trade Loans",
            "Patriot Express Loans",
            "Export Express",
            "Contract CAPLine",
            "Seasonal CAPLine",
            "Working Capital CAPLine",
            "Rural Loan Initiative"
        ]
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

    congressional_district = st.number_input(
        "Congressional District",
        min_value=0.0,
        value=25.0,
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
        max_value=2030,
        value=2007,
        step=1
    )

    disbursement_month = st.number_input(
        "Disbursement Month",
        min_value=1,
        max_value=12,
        value=7,
        step=1
    )

    disbursement_year = st.number_input(
        "Disbursement Year",
        min_value=1999,
        max_value=2030,
        value=2007,
        step=1
    )

    disbursement_delay = st.number_input(
        "Disbursement Delay (Days)",
        min_value=0.0,
        value=30.0,
        step=1.0
    )


# --------------------------------------------------
# Business Information
# --------------------------------------------------

st.header("Business Information")

col1, col2, col3 = st.columns(3)


with col1:

    business_type = st.selectbox(
        "Business Type",
        [
            "CORPORATION",
            "INDIVIDUAL",
            "PARTNERSHIP"
        ]
    )

    business_age = st.selectbox(
        "Business Age",
        [
            "Existing, 5 or more years",
            "Less than 3 years old but at least 2",
            "Less than 4 years old but at least 3",
            "Less than 5 years old but at least 4",
            "New, Less than 1 Year old",
            "Startup, Loan Funds will Open Business",
            "Unanswered"
        ]
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

    sold_secondary = st.selectbox(
        "Sold in Secondary Market",
        ["N", "Y", "Unknown"]
    )

    project_state = st.selectbox(
        "Project State",
        [
            "AL", "CA", "FL", "GA", "IL", "NY",
            "NC", "OH", "PA", "TX", "VA", "WA", "Other"
        ]
    )


with col3:

    naics_code = st.text_input(
        "NAICS Code",
        value="722511"
    )

    naics_description = st.text_input(
        "NAICS Description",
        value="Full-Service Restaurants"
    )

    project_county = st.text_input(
        "Project County",
        value="MIAMI-DADE"
    )

    district_office = st.text_input(
        "SBA District Office",
        value="SOUTH FLORIDA DISTRICT OFFICE"
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "🔍 Predict Default Risk",
    use_container_width=True
):

    # Automatically create TermGroup

    term_group = get_term_group(term)


    # Create input DataFrame

    input_data = pd.DataFrame([{

        "BorrState": borr_state,
        "GrossApproval": gross_approval,
        "SBAGuaranteedApproval": sba_guaranteed,
        "ApprovalFY": approval_fy,
        "ProcessingMethod": processing_method,
        "TermInMonths": term,
        "NaicsCode": naics_code,
        "NaicsDescription": naics_description,
        "ProjectCounty": project_county,
        "ProjectState": project_state,
        "SBADistrictOffice": district_office,
        "CongressionalDistrict": congressional_district,
        "BusinessType": business_type,
        "BusinessAge": business_age,
        "RevolverStatus": revolver_status,
        "JobsSupported": jobs_supported,
        "CollateralInd": collateral,
        "SoldSecMrktInd": sold_secondary,
        "ApprovalMonth": approval_month,
        "ApprovalYear": approval_year,
        "DisbursementMonth": disbursement_month,
        "DisbursementYear": disbursement_year,
        "DisbursementDelayDays": disbursement_delay,
        "TermGroup": term_group

    }])


    # --------------------------------------------------
    # Preprocess Input
    # --------------------------------------------------

    processed_input = preprocessor.transform(input_data)


    # --------------------------------------------------
    # Predict Probability
    # --------------------------------------------------

    probability = model.predict_proba(
        processed_input
    )[0][1]


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    prediction = int(probability >= 0.50)


    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.divider()

    st.header("Prediction Result")

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Default Probability",
            f"{probability * 100:.2f}%"
        )


    with col2:

        if prediction == 1:

            st.error("⚠️ Predicted: DEFAULT")

        else:

            st.success("✅ Predicted: NO DEFAULT")


    st.progress(float(probability))

    st.caption(
        "Prediction threshold: 0.50"
    )