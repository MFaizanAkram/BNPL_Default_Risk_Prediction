#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
import streamlit as st
from PIL import Image
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
#---------------------------------------------------------------------------------------------#
# Load model assets (load once)
model = joblib.load("xgb_model.pkl")
feature_names = joblib.load("feature_names.pkl")
CUTOFF = joblib.load("cutoff.pkl")
# SHAP explainer (load once)
explainer = shap.TreeExplainer(model)
#---------------------------------------------------------------------------------------------#
def bnpl_prediction_ui():
    st.subheader("Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 70)
        monthly_income = st.number_input("Monthly Income")
        purchase_amount = st.number_input("Purchase Amount")
        installments = st.selectbox("Installments", [3, 6, 9, 12])
        employment = st.selectbox(
            "Employment Status",
            ["Salaried", "Self-Employed", "Unemployed"]
        )

    with col2:
        avg_overdue_days = st.number_input("Avg payment delay days")
        missed_payments = st.number_input("Missed Payments")
        previous_bnpl_loans = st.number_input("Previous BNPL Loans")

    if st.button("Predict Risk"):
        employment_self = 1 if employment == "Self-Employed" else 0
        employment_unemp = 1 if employment == "Unemployed" else 0

        EPS = 1e-6
        safe_purchase = max(purchase_amount, EPS)
        safe_income = max(monthly_income, EPS)

        income_purchase_ratio = safe_income / safe_purchase
        installment_amount = purchase_amount / installments
        installment_to_income = installment_amount / safe_income
        payment_risk_score = (
            0.6 * avg_overdue_days + 0.4 * missed_payments
        )

        input_data = pd.DataFrame([{
            "age": age,
            "monthly_income": monthly_income,
            "purchase_amount": purchase_amount,
            "installments": installments,
            "previous_bnpl_loans": previous_bnpl_loans,
            "avg_overdue_days": avg_overdue_days,
            "missed_payments": missed_payments,
            "employment_status_Self-Employed": employment_self,
            "employment_status_Unemployed": employment_unemp,
            "income_purchase_ratio": income_purchase_ratio,
            "installment_amount": installment_amount,
            "installment_to_income": installment_to_income,
            "payment_risk_score": payment_risk_score
        }])

        input_data = input_data[feature_names]

        prob_default = model.predict_proba(input_data)[0][1]
        shap_values = explainer.shap_values(input_data)
        # ---- Risk Bands ----
        if prob_default < 0.15:
            risk_level = "LOW RISK"
            decision = "APPROVE"
        elif prob_default < 0.30:
            risk_level = "MEDIUM RISK"
            decision = "CONDITIONAL APPROVAL"
        else:
            risk_level = "HIGH RISK"
            decision = "REJECT"


        st.subheader("Risk Assessment Result")

        st.metric("Probability of Default", f"{prob_default:.2%}")
        st.metric("Risk Category", risk_level)

        if decision == "APPROVE":
            st.success("Decision: APPROVED")
        elif decision == "CONDITIONAL APPROVAL":
            st.warning("Decision: CONDITIONAL APPROVAL")
        else:
            st.error("Decision: REJECTED")

# ---- Conditional Approval Rules ----
        st.subheader("Conditional Approval Rules")
        if decision == "CONDITIONAL APPROVAL":
            st.subheader("Recommended Conditions")

        if installment_to_income > 0.30:
            st.write("- Reduce number of installments (max 6)")

        if purchase_amount > monthly_income * 0.5:
            st.write("- Lower purchase amount")

        st.write("- Require 20% upfront payment")

# ---- Decision Explanation ----
        st.subheader("Decision Explanation")

        reasons = []

        if installment_to_income > 0.35:
            reasons.append("Installment amount is high relative to income")

        if avg_overdue_days > 10:
            reasons.append("Frequent payment delays in past behavior")

        if missed_payments > 2:
            reasons.append("Multiple missed payments observed")

        if previous_bnpl_loans > 3:
            reasons.append("High number of previous BNPL loans")

        if reasons:
            for r in reasons:
                st.write("•", r)
        else:
            st.write("• No major risk factors detected")

# ---- SHAP Explanation ----
        st.subheader("Model Explanation (SHAP)")

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_values[0]})

        shap_df["Impact"] = shap_df["SHAP Value"].apply(
            lambda x: "Increases Risk" if x > 0 else "Decreases Risk")

        shap_df = shap_df.reindex(shap_df["SHAP Value"].abs().sort_values(ascending=False).index)

        st.dataframe(shap_df.head(6), use_container_width=True)


#---------------------------------------------------------------------------------------------#
# Streamlit App: Theory Subject Page
def theory_subject_page():
    st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background: linear-gradient(to right, #f7f9fc, #eef2f7); /* Softer gradient */
        color: #333333;}
    .stButton>button {
        background-color: #f1f1f1;  /* Light Grey for Buttons */
        color: #333333;  /* Dark Text for Visibility */
        border-radius: 12px;
        padding: 10px;
        font-size: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;}
    .stButton>button:hover {
        background-color: #e0e0e0;  /* Slightly Darker Light Grey */
        transform: scale(1.05);  /* Slight zoom effect */}
    .stMarkdown>p {
        text-align: justify;
        color: #333;
        font-size: 16px;}
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);}
    .stHeader>h1 {
        color: #eab676;  /* Soft Beige */
        font-weight: bold;
        font-size: 30px;
        text-align: center;}
    .stSubheader>h2 {
        font-size: 22px;
        color: #388e3c;  /* Soft Green */}
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;}
    .results {
        transition: background-color 0.3s ease;}
    .results:hover {
        background-color: #fff3e0; /* Soft light beige */}
    .stMarkdown ul {
        list-style-type: none;
        padding-left: 0;}
    .stMarkdown li {
        padding: 5px 0;}
    .stTextInput>label {
        font-size: 16px;
        color: #eab676; /* Soft Beige */}
    .stIcon {
        color: black !important;  /* Set all icons color to black */}
    </style>
    """, unsafe_allow_html=True)
    bnpl_prediction_ui()
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
    st.sidebar.image("logo.png", use_container_width=True)
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#    

#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
    st.sidebar.title("About me:")
    st.sidebar.markdown(
        """
        - **Name:** Muhammad Faizan Akram  
        - **Reg No:** FA23-BBD-090 (5A)
        - **Email:** mfakram28@gmail.com
        """)
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def main():
    st.set_page_config(
        page_title="BNPL Default Risk Prediction",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <h1 style="text-align: center; color: #333333; font-weight: bold; font-size: 33px;">
            Buy Now Pay Later (BNPL) Default Risk Prediction
        </h1>
        """,
        unsafe_allow_html=True
    )

    # Directly load the single page
    theory_subject_page()



if __name__ == "__main__":
    main()
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#