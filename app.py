# app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -----------------------------
# Load trained pipeline
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/best_churn_model.pkl")

model = load_model()

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].map({1: 'Yes', 0: 'No'})
    return df

data = load_data()

# -----------------------------
# Streamlit app title
# -----------------------------
st.title("Telco Customer Churn App")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["Prediction", "Data"])

# -----------------------------
# Prediction Tab
# -----------------------------
with tab1:
    st.markdown("Enter customer information below to predict churn:")

    def user_input_features():
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
        OnlineBackup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
        DeviceProtection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
        TechSupport = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
        StreamingTV = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                                        "Bank transfer (automatic)", "Credit card (automatic)"])
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, value=500.0)

        data = {
            'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'tenure': tenure,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    if st.button("Predict Churn"):
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else None

        st.subheader("Prediction")
        st.write("Churn: **Yes**" if prediction == 1 else "Churn: **No**")

        if prediction_proba is not None:
            st.write(f"Probability of churn: {prediction_proba:.2%}")

            # -----------------------------
            # Plotly bar chart
            # -----------------------------
            pred_df = pd.DataFrame({
                'Churn': ['No', 'Yes'],
                'Probability': [1 - prediction_proba, prediction_proba]
            })

            fig = px.bar(
                pred_df, x='Churn', y='Probability', text=pred_df['Probability'].apply(lambda x: f"{x:.2%}"),
                color='Churn', color_discrete_map={'No':'green', 'Yes':'red'},
                labels={'Probability':'Probability'}, height=400
            )
            fig.update_layout(yaxis=dict(range=[0,1]))
            st.plotly_chart(fig)

# -----------------------------
# Data Tab
# -----------------------------
with tab2:
    st.subheader("Customer Data")
    st.dataframe(data)
