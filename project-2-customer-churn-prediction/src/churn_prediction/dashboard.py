import joblib
import pandas as pd
import shap
import streamlit as st

from churn_prediction.config import ARTIFACTS_DIR


def load_artifacts():
    pipeline = joblib.load(ARTIFACTS_DIR / "feature_pipeline.joblib")
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    feature_columns = joblib.load(ARTIFACTS_DIR / "feature_columns.joblib")
    threshold = joblib.load(ARTIFACTS_DIR / "threshold.joblib")
    return pipeline, model, feature_columns, threshold


def get_feature_names(pipeline) -> list[str]:
    raw_names = pipeline.named_steps["column_transform"].get_feature_names_out()
    return [name.split("__", 1)[-1] for name in raw_names]


def predict_one(pipeline, model, threshold: float, input_dict: dict) -> dict:
    row = pd.DataFrame([input_dict])
    X = pipeline.transform(row)
    proba = float(model.predict_proba(X)[:, 1][0])
    return {
        "probability": proba,
        "prediction": "Churn" if proba >= threshold else "No churn",
        "X": X,
    }


def explain_one(model, X, feature_names: list[str], top_n: int = 10) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    contributions = pd.DataFrame({"feature": feature_names, "contribution": shap_values.values[0]})
    contributions["abs_contribution"] = contributions["contribution"].abs()
    return contributions.sort_values("abs_contribution", ascending=False).head(top_n)


def run() -> None:
    st.set_page_config(page_title="Customer churn predictor", layout="centered")
    st.title("Customer churn predictor")
    st.caption("Telco Customer Churn -- tuned LightGBM, cost-optimal decision threshold")

    pipeline, model, feature_columns, threshold = load_artifacts()

    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
        with col2:
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

        monthly_charges = st.slider("Monthly charges ($)", 18.0, 120.0, 65.0)
        submitted = st.form_submit_button("Predict churn risk")

    if submitted:
        total_charges = monthly_charges * max(tenure, 1)
        input_dict = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": str(total_charges),
        }
        result = predict_one(pipeline, model, threshold, input_dict)

        st.metric("Churn probability", f"{result['probability']:.1%}")
        if result["prediction"] == "Churn":
            st.error(f"Predicted: likely to churn (decision threshold={threshold:.2f})")
        else:
            st.success(f"Predicted: likely to stay (decision threshold={threshold:.2f})")

        feature_names = get_feature_names(pipeline)
        contributions = explain_one(model, result["X"], feature_names)
        st.subheader("Why this prediction (SHAP)")
        st.caption("Positive values push the prediction toward churn; negative values push toward staying.")
        st.bar_chart(contributions.set_index("feature")["contribution"])


if __name__ == "__main__":
    run()
