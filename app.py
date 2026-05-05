import streamlit as st
import pandas as pd
from joblib import load

# Load files
model = load("model_logit.joblib")
scaler = load("scaler.joblib")
feature_list = load("feature_list.joblib")

# Columns to scale
scaler_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

st.title("📞 Telecom Churn Predictor")

# Inputs
tenure = st.slider("Tenure", 0, 72, 12)
monthly = st.slider("Monthly Charges", 0.0, 120.0, 65.0)
total = st.slider("Total Charges", 0.0, 9000.0, 1000.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

if st.button("Predict"):

    data = {col: 0 for col in feature_list}

    data["tenure"] = tenure
    data["MonthlyCharges"] = monthly
    data["TotalCharges"] = total

    if contract == "One year":
        data["Contract_One year"] = 1
    elif contract == "Two year":
        data["Contract_Two year"] = 1

    if internet == "Fiber optic":
        data["InternetService_Fiber optic"] = 1
    elif internet == "No":
        data["InternetService_No"] = 1

    if payment == "Electronic check":
        data["PaymentMethod_Electronic check"] = 1
    elif payment == "Mailed check":
        data["PaymentMethod_Mailed check"] = 1
    elif payment == "Credit card (automatic)":
        data["PaymentMethod_Credit card (automatic)"] = 1
    elif payment == "Bank transfer (automatic)":
        data["PaymentMethod_Bank transfer (automatic)"] = 1

    df = pd.DataFrame([data])[feature_list]

    # Scale
    df[scaler_columns] = scaler.transform(df[scaler_columns])

    # Predict
    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    # Output
    if pred == 1:
        st.error(f"Customer will CHURN ({prob:.2f})")
    else:
        st.success(f"Customer will NOT churn ({prob:.2f})")