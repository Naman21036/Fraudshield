import streamlit as st
import requests

# -------------------------------
# Config
# -------------------------------
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------
# Header
# -------------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>FraudShield</h1>
    <p style='text-align: center; color: gray;'>
    Real time transaction fraud detection system
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Transaction Inputs")

Time = st.sidebar.number_input(
    "Time since first transaction (seconds)",
    min_value=0.01,
    value=10000.0
)

Amount = st.sidebar.number_input(
    "Transaction Amount",
    min_value=0.01,
    value=100.0
)

st.sidebar.markdown("### PCA Features")

V = {}
for i in range(1, 29):
    V[f"V{i}"] = st.sidebar.number_input(
        f"V{i}",
        value=0.0,
        step=0.01
    )

predict_btn = st.sidebar.button("Run Fraud Detection")

# -------------------------------
# Main Panel
# -------------------------------
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Transaction Summary")
    st.write(f"**Time:** {Time} seconds")
    st.write(f"**Amount:** ₹ {Amount:.2f}")

    st.markdown("#### PCA Snapshot")
    st.dataframe(
        {k: [v] for k, v in V.items()},
        use_container_width=True
    )

with col2:
    st.subheader("Prediction Result")

    if predict_btn:
        payload = {
            "Time": Time,
            "Amount": Amount,
            **V
        }

        try:
            with st.spinner("Contacting FraudShield API..."):
                response = requests.post(API_URL, json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()

                prediction = result["fraud_prediction"]
                probability = result.get("fraud_probability")

                if prediction == 1:
                    st.error("Fraudulent Transaction Detected")
                else:
                    st.success("Transaction Appears Legitimate")

                if probability is not None:
                    st.metric(
                        label="Fraud Probability",
                        value=f"{probability:.4f}"
                    )

                    st.progress(min(int(probability * 100), 100))

            else:
                st.error("API Error")
                st.json(response.json())

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI server")
        except Exception as e:
            st.error(str(e))

    else:
        st.info("Enter transaction details and click **Run Fraud Detection**")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption(
    "FraudShield • Machine Learning based fraud detection system"
)
st.caption("Developed by Naman Gupta")