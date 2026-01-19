import streamlit as st
import requests
import pandas as pd

# --------------------------------------------------
# Config
# --------------------------------------------------
API_BASE_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{API_BASE_URL}/predict"

st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# Dark mode styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">FraudShield</h1>
    <p style="text-align:center; color: #9CA3AF;">
    Real time fraud detection system
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --------------------------------------------------
# API Health Indicator
# --------------------------------------------------
with st.sidebar:
    st.subheader("API Status")
    try:
        health = requests.get(API_BASE_URL, timeout=3)
        if health.status_code == 200:
            st.success("API is running")
        else:
            st.warning("API reachable but unhealthy")
    except:
        st.error("API not reachable")

    st.markdown("---")

# --------------------------------------------------
# Threshold slider
# --------------------------------------------------
with st.sidebar:
    st.subheader("Decision Threshold")
    threshold = st.slider(
        "Fraud Probability Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
with st.sidebar:
    st.subheader("Transaction Inputs")

    Time = st.number_input("Time (seconds)", min_value=0.01, value=10000.0)
    Amount = st.number_input("Amount", min_value=0.01, value=100.0)

    st.markdown("### PCA Features")
    V = {}
    for i in range(1, 29):
        V[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.01)

    run_single = st.button("Run Single Prediction")

# --------------------------------------------------
# Batch CSV Upload
# --------------------------------------------------
st.markdown("## Batch Prediction")
uploaded_file = st.file_uploader(
    "Upload CSV file (same schema as API input)",
    type=["csv"]
)

if uploaded_file:
    df_batch = pd.read_csv(uploaded_file)
    st.dataframe(df_batch.head())

    if st.button("Run Batch Prediction"):
        results = []

        for _, row in df_batch.iterrows():
            payload = row.to_dict()
            try:
                r = requests.post(PREDICT_URL, json=payload, timeout=5)
                if r.status_code == 200:
                    res = r.json()
                    res["risk_band"] = (
                        "High" if res["fraud_probability"] >= threshold
                        else "Medium" if res["fraud_probability"] >= threshold * 0.6
                        else "Low"
                    )
                    results.append(res)
                else:
                    results.append({"error": "API error"})
            except:
                results.append({"error": "Connection failed"})

        st.markdown("### Batch Results")
        st.dataframe(pd.DataFrame(results))

st.markdown("---")

# --------------------------------------------------
# Main Prediction Panel
# --------------------------------------------------
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Transaction Summary")
    st.write(f"Time: {Time}")
    st.write(f"Amount: ₹{Amount:.2f}")

    st.markdown("PCA Snapshot")
    st.dataframe({k: [v] for k, v in V.items()}, use_container_width=True)

with col2:
    st.subheader("Prediction Result")

    if run_single:
        payload = {
            "Time": Time,
            "Amount": Amount,
            **V
        }

        try:
            with st.spinner("Calling FraudShield API..."):
                response = requests.post(PREDICT_URL, json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()
                prob = result["fraud_probability"]

                # Threshold based decision
                prediction = 1 if prob >= threshold else 0

                # Risk bands
                if prob >= threshold:
                    st.error("High Risk Fraud")
                    risk = "High"
                elif prob >= threshold * 0.6:
                    st.warning("Medium Risk Transaction")
                    risk = "Medium"
                else:
                    st.success("Low Risk Transaction")
                    risk = "Low"

                st.metric("Fraud Probability", f"{prob:.4f}")
                st.progress(min(int(prob * 100), 100))

                # Model confidence explanation
                st.markdown("### Model Confidence Explanation")
                st.write(
                    f"""
                    The model estimates a **{prob:.2%} probability** of fraud.
                    
                    Based on the selected threshold (**{threshold:.2f}**),
                    this transaction is classified as **{risk} risk**.
                    
                    Adjusting the threshold allows you to trade off between
                    false positives and false negatives.
                    """
                )

            else:
                st.error("API error")
                st.json(response.json())

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI server")
        except Exception as e:
            st.error(str(e))

    else:
        st.info("Enter details and click **Run Single Prediction**")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("FraudShield • End to end ML system with API and UI")
st.caption("Developed by Naman Gupta")