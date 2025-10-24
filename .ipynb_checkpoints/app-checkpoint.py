import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# ------------------ Sidebar Branding ------------------
st.sidebar.image("https://vectorseek.com/wp-content/uploads/2023/08/Streamlit-Logo-Vector.svg-.png", width=150)
st.sidebar.markdown("### Powered by Deepak")
st.sidebar.markdown("Smart Loan Approval | AI | Streamlit")
st.sidebar.markdown("---")
st.sidebar.markdown("📬 [Connect on LinkedIn](https://www.linkedin.com/in/deepak-khadka-78869a221)")



# ------------------ Page Setup ------------------
st.set_page_config(page_title="Smart Loan Approval", page_icon="🏦", layout="centered")
st.title("🏦 Smart Loan Approval System")
st.markdown("Predict loan approval based on applicant details.")

# ------------------ Sample Model Setup ------------------
# Dummy training data
train_data = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Male', 'Female'],
    'Married': ['Yes', 'No', 'Yes', 'Yes'],
    'Education': ['Graduate', 'Not Graduate', 'Graduate', 'Graduate'],
    'ApplicantIncome': [5000, 3000, 6000, 4000],
    'LoanAmount': [150, 100, 200, 120],
    'Credit_History': [1, 0, 1, 1],
    'Loan_Status': [1, 0, 1, 1]
})

# Encode categorical columns
encoders = {}
for col in ['Gender', 'Married', 'Education']:
    enc = LabelEncoder()
    train_data[col] = enc.fit_transform(train_data[col])
    encoders[col] = enc

# Features and target
X = train_data.drop('Loan_Status', axis=1)
y = train_data['Loan_Status']

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# ------------------ Single Prediction ------------------
st.subheader("🔍 Single Applicant Prediction")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", encoders['Gender'].classes_)
    married = st.selectbox("Married", encoders['Married'].classes_)
    education = st.selectbox("Education", encoders['Education'].classes_)
with col2:
    income = st.slider("Applicant Income", 2000, 10000, 5000)
    loan_amount = st.slider("Loan Amount", 50, 500, 150)
    credit_history = st.selectbox("Credit History", [0, 1])

if st.button("Predict Loan Status"):
    input_data = pd.DataFrame([{
        'Gender': encoders['Gender'].transform([gender])[0],
        'Married': encoders['Married'].transform([married])[0],
        'Education': encoders['Education'].transform([education])[0],
        'ApplicantIncome': income,
        'LoanAmount': loan_amount,
        'Credit_History': credit_history
    }])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][prediction]

    result = "✅ Approved" if prediction == 1 else "❌ Rejected"
    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {proba:.2f}")

# ------------------ Batch Prediction ------------------
st.markdown("---")
st.subheader("📦 Batch Prediction from CSV")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="batch_upload")

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)

        # Encode categorical columns
        for col in ['Gender', 'Married', 'Education']:
            batch_df[col] = encoders[col].transform(batch_df[col])

        # Scale numeric features
        batch_scaled = scaler.transform(batch_df)

        # Predict
        batch_preds = model.predict(batch_scaled)
        batch_probas = model.predict_proba(batch_scaled)

        batch_df['Prediction'] = ['Approved' if p == 1 else 'Rejected' for p in batch_preds]
        batch_df['Confidence'] = [f"{max(proba):.2f}" for proba in batch_probas]

        st.write("🔍 Prediction Results:")
        for i, row in batch_df.iterrows():
            st.text(f"Row {i+1}: Prediction = {row['Prediction']}, Confidence = {row['Confidence']}")

    except pd.errors.EmptyDataError:
        st.error("❌ Uploaded file is empty or invalid. Please check your CSV format.")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
