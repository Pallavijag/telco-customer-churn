# Telco Customer Churn Prediction

An end-to-end machine learning system to predict telecom customer churn using Logistic Regression and Random Forest, built with a full Scikit-learn pipeline and deployed through a Streamlit web application.

## Overview

Customer churn directly impacts recurring revenue in telecom businesses.  
This project develops a production-style churn prediction system that:

- Cleans and preprocesses customer data
- Compares Logistic Regression and Random Forest models
- Uses a full Scikit-learn Pipeline (encoding + scaling + model)
- Performs cross-validation and threshold optimization
- Evaluates using ROC-AUC and classification metrics
- Deploys a real-time Streamlit prediction interface
- Exports high-risk customers for retention strategy

## Model Development

### Models Evaluated
- Logistic Regression
- Random Forest Classifier

Both models were evaluated using Stratified 5-Fold Cross-Validation.

Logistic Regression was selected for deployment due to:
- Comparable ROC-AUC performance
- Better interpretability
- Simpler inference pipeline
- Lower computational complexity

## Model Performance

- Cross-Validation ROC-AUC: ~0.84  
- Test ROC-AUC: ~0.83  
- Accuracy: ~78%  
- Recall (Churn class): ~0.68  
- Optimized Decision Threshold: 0.4  

Threshold was tuned from default 0.5 → 0.4 to improve churn recall while maintaining reasonable precision, aligning with business sensitivity toward churn risk.

## Feature Insights

Key churn drivers identified:

- Low Tenure
- Month-to-month Contracts
- Fiber Optic Internet Service
- Electronic Check Payment Method
- No Tech Support
- No Online Security

Feature importance and coefficient analysis were performed to interpret model behavior.

## Project Structure

TELCO_CUSTOMER_CHURN/
│
├── app.py
├── train_model.py
├── test_model.py
│
├── models/
│   └── churn_pipeline.joblib
│
├── data/
│   ├── cleaned_telco_data.csv
│   ├── at_risk_customers.csv
│   └── risk_dashboard.csv
│
├── requirements.txt
├── .gitignore
└── README.md

## Installation & Setup

### Clone Repository

git clone https://github.com/yourusername/telco-customer-churn.git  
cd telco-customer-churn  

### Create Virtual Environment

python -m venv .venv  

Activate:

Windows:
.venv\Scripts\activate  

Mac/Linux:
source .venv/bin/activate  

### Install Dependencies

pip install -r requirements.txt  

## Train the Model

python train_model.py  

Generates:

models/churn_pipeline.joblib  

## Run the Streamlit App

streamlit run app.py  

App Features:
- Interactive churn simulation
- Real-time probability scoring
- Threshold-based decision logic
- Clean UI for business-style input

## Risk Export

The system generates:

- at_risk_customers.csv
- risk_dashboard.csv

These files support:

- Targeted retention campaigns
- Business dashboard integration
- Revenue risk forecasting

## Business Impact

This model enables telecom providers to:

- Identify high-risk customers proactively
- Reduce churn-related revenue loss
- Optimize retention resource allocation
- Improve Customer Lifetime Value (CLTV)

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib
- Streamlit

## Future Improvements

- Gradient Boosting (XGBoost / LightGBM)
- Hyperparameter Optimization (GridSearchCV / RandomizedSearchCV)
- SHAP-based Model Explainability
- Model Monitoring & Drift Detection
- Docker Containerization
- Cloud Deployment (Streamlit Cloud / AWS)

---

End-to-end churn prediction system built with production-style ML pipeline, model comparison, threshold optimization, and deployable architecture.