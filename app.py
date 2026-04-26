import streamlit as st
import pandas as pd
from joblib import load

model = load("model_logit.joblib")
scaler = load("scaler.joblib")
feature_list = load("feature_list.joblib")

st.title("📞 Telecom Churn Predictor")

st.write("Enter customer details to predict churn")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 0.0, 120.0, 65.0)
total_charges = st.slider("Total Charges", 0.0, 9000.0, 1000.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                           "Bank transfer (automatic)",
                                           "Credit card (automatic)"])


if st.button("Predict"):

    
    data = {col: 0 for col in feature_list}

   
    data["tenure"] = tenure
    data["MonthlyCharges"] = monthly_charges
    data["TotalCharges"] = total_charges

    
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

   
    df = pd.DataFrame([data])[feature_list]


    df_scaled = df.copy()
    df_scaled[scaler.feature_names_in_] = scaler.transform(df_scaled[scaler.feature_names_in_])

   
    prob = model.predict_proba(df_scaled)[0][1]
    pred = model.predict(df_scaled)[0]

   
    st.write("### Result:")

    if pred == 1:
        st.error(f"⚠ Customer will CHURN (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer will NOT churn (Probability: {prob:.2f})")