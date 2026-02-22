import streamlit as st
import pandas as pd
import joblib

# 1. Load the "Brain" (Model and Preprocessor)
# These files must be in the same folder as this script
try:
    model = joblib.load('best_churn_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
except FileNotFoundError:
    st.error("Error: Model files not found. Please run 'churn_main.py' first to generate .pkl files.")
    st.stop()

st.title("🚀 Customer Churn Predictor")
st.markdown("Enter customer details below to predict if they will leave the company.")

# 2. Create Input Fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    st.subheader("Service Details")
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col2:
    st.subheader("Contract & Billing")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)

    st.subheader("Tech Support")
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

# 3. Collect Data into a DataFrame
input_data = pd.DataFrame({
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges]
})

# 4. Predict
if st.button("Predict Churn"):
    # A. Transform the data using the saved preprocessor
    # This automatically handles OneHotEncoding and Scaling!
    try:
        processed_data = preprocessor.transform(input_data)

        # B. Make Prediction
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[0][1] # Probability of Churn (Class 1)

        st.divider()

        # C. Display Results
        if prediction[0] == 1:
            st.error(f"⚠️ HIGH RISK: This customer is likely to CHURN.")
            st.write(f"**Confidence:** {probability:.2%}")
        else:
            st.success(f"✅ SAFE: This customer is likely to STAY.")
            st.write(f"**Churn Risk:** {probability:.2%}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")