# SBA Loan Default Predictor

A machine learning project that predicts the probability of default for an SBA-backed small-business loan using information available at the time of loan approval.

The project covers the complete ML workflow:

- Data understanding and cleaning
- Feature engineering
- Exploratory analysis
- Time-based train/test splitting
- Machine learning model training
- Probability calibration
- Model evaluation
- Streamlit deployment

## Problem Statement

Loan default prediction is a binary classification problem.

Given information about a small-business loan and the business at the time of approval, the model estimates the probability that the loan will eventually default.

The goal is to build a model that can identify historical patterns associated with loan default while avoiding information that would only become available after the loan was approved.

## Dataset

The project uses the U.S. Small Business Administration (SBA) 7(a)/504 FOIA loan dataset.

Official source:

https://data.sba.gov/dataset/7a-504-foia

The dataset contains historical SBA-backed loan records.

For this project:

- `P I F` (Paid In Full) → Default = 0
- `CHGOFF` (Charged Off) → Default = 1
- `CANCLD` (Cancelled) → excluded
- `EXEMPT` → excluded

The raw dataset is intentionally not included in this repository because of its large size.

## Prediction Point

The prediction is designed around information available at the time of loan approval.

Features representing information that occurs after approval, such as disbursement-related or secondary-market information, were excluded.

This prevents future information from leaking into the model.

## Feature Engineering

The final model uses 16 features.

### Numerical Features

- GrossApproval
- SBAGuaranteedApproval
- TermInMonths
- JobsSupported
- ApprovalMonth
- ApprovalYear

### Categorical Features

- BorrState
- ProcessingMethod
- NaicsSector
- ProjectState
- SBADistrictOffice
- BusinessType
- BusinessAge
- RevolverStatus
- CollateralInd
- TermGroup

NAICS codes are converted into broader 2-digit industry sectors.

Loan terms are also grouped into meaningful ranges using `TermGroup`.

## Model

The final model is a Random Forest classifier.

Configuration:

- 100 trees
- Maximum depth: 12
- Minimum samples per leaf: 5
- Balanced class weights
- Random state: 42

Categorical features are handled using one-hot encoding.

Numerical features use median imputation and standard scaling.

## Time-Based Evaluation

Instead of randomly splitting the historical data, the project uses chronological splits to better represent how the model would behave when predicting future loans.

### Training

1999–2005

### Calibration

2006

An Isotonic Regression calibrator was trained using the separate 2006 period.

### Final Test

2007–2009

This keeps the final evaluation period completely separate from both model training and probability calibration.

## Final Test Performance

Performance on the 2007–2009 test period:

| Metric | Score |
|---|---:|
| Accuracy | 0.8062 |
| Precision | 0.6609 |
| Recall | 0.7265 |
| F1 Score | 0.6921 |
| ROC-AUC | 0.8676 |
| PR-AUC | 0.6832 |

The classification threshold used by the application is 50%.

## Performance by Year

| Year | Default Rate | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 2007 | 37.34% | 0.739 | 0.743 | 0.741 | 0.881 |
| 2008 | 26.64% | 0.635 | 0.699 | 0.665 | 0.864 |
| 2009 | 13.79% | 0.368 | 0.685 | 0.479 | 0.802 |

Performance varies across years, which highlights the effect of changes in the underlying loan population and default rate over time.

## Probability Calibration

The raw Random Forest probabilities showed substantial overprediction when evaluated on later years.

To address this, an Isotonic Regression calibrator was trained on the separate 2006 calibration period.

The Streamlit application displays the calibrated probability rather than the raw Random Forest probability.

Calibration improves the interpretation of the predicted probabilities, but the probabilities should not be treated as perfectly accurate estimates for every future population.

## Streamlit Application

The project includes an interactive Streamlit application.

Users can enter:

- Loan amount
- SBA guaranteed amount
- Loan term
- Jobs supported
- Approval month/year
- Borrower state
- Project state
- Processing method
- Business type
- Business age
- Revolver status
- Collateral information
- SBA district office
- NAICS code

The application then displays:

- Estimated default probability
- Predicted class
- Classification threshold

## Project Structure

```text
sba-loan-default/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── raw/              # Raw dataset, not committed
│
├── models/
│   ├── final_random_forest.pkl
│   ├── final_preprocessor.pkl
│   └── final_calibrator.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   └── 02_modeling.ipynb
│
└── src/