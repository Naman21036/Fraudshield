import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "https://fraudshield-fraud-transaction.onrender.com"
PREDICT_URL = f"{API_BASE_URL}/predict"

st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide",
)

if "result" not in st.session_state:
    st.session_state.result = None
if "count" not in st.session_state:
    st.session_state.count = 0

st.markdown("""<style>
.stApp { background-color: #060e1c; color: #c8d4e8; }

section[data-testid="stSidebar"] {
    background-color: #07142a;
    border-right: 1px solid #0e2040;
}
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1080px;
}

/* Header */
.fs-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #0e2040;
    margin-bottom: 1.6rem;
}
.fs-title { font-size: 1.4rem; font-weight: 700; color: #dce8ff; letter-spacing: -0.4px; }
.fs-sub   { font-size: 0.68rem; color: #2a4268; text-transform: uppercase; letter-spacing: 0.12em; }

/* Labels */
.fs-label {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #2a4268;
    margin-bottom: 0.55rem;
    margin-top: 1rem;
}

/* Cards */
.fs-card {
    border-radius: 8px;
    padding: 1.8rem 2rem;
    min-height: 320px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.card-idle { background: #091728; border: 1px solid #0e2040; }
.card-low  { background: #030f07; border: 1px solid #14532d; }
.card-mid  { background: #0e0900; border: 1px solid #6b3800; }
.card-high { background: #0e0303; border: 1px solid #7f1d1d; }

.fs-prob-unit {
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #2a4268;
    margin-bottom: 0.2rem;
}
.fs-prob {
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
    font-variant-numeric: tabular-nums;
    letter-spacing: -1px;
    margin-bottom: 0.9rem;
}
.p-idle { color: #0f1e30; }
.p-low  { color: #22c55e; }
.p-mid  { color: #f59e0b; }
.p-high { color: #ef4444; }

.fs-verdict { font-size: 1.05rem; font-weight: 600; line-height: 1.3; }
.v-idle { color: #1a2d47; }
.v-low  { color: #16a34a; }
.v-mid  { color: #b45309; }
.v-high { color: #dc2626; }

.fs-meta {
    font-size: 0.68rem;
    color: #1e3352;
    margin-top: 0.6rem;
    line-height: 1.6;
}
.fs-count {
    font-size: 0.62rem;
    color: #1e3352;
    margin-top: 1.2rem;
    padding-top: 0.8rem;
    border-top: 1px solid #0e2040;
    letter-spacing: 0.04em;
}

/* Status */
.fs-status { font-size: 0.75rem; color: #3d5a80; }

/* Fix red tab indicator → blue */
[data-baseweb="tab-highlight"] {
    background-color: #2a5ccc !important;
}
[data-baseweb="tab"] {
    color: #2a4268 !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #c8d8f4 !important;
}

/* Buttons */
.stButton > button {
    background-color: #0d2b58 !important;
    color: #93b8f0 !important;
    border: 1px solid #173f7a !important;
    border-radius: 5px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background-color: #112f63 !important;
    border-color: #1f50a0 !important;
}

/* Expander */
div[data-testid="stExpander"] {
    border: 1px solid #0e2040 !important;
    border-radius: 6px !important;
}

/* Progress bar track */
div[data-testid="stProgress"] > div {
    background-color: #091728 !important;
    border-radius: 4px !important;
}
</style>""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fs-header">
    <span class="fs-title">🛡 FraudShield</span>
    <span class="fs-sub">Transaction Risk Analysis</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="fs-label">System</div>', unsafe_allow_html=True)
    try:
        health = requests.get(API_BASE_URL, timeout=4)
        status = "🟢 API online" if health.status_code == 200 else "🟡 API degraded"
    except Exception:
        status = "🔴 API unreachable"
    st.markdown(f'<div class="fs-status">{status}</div>', unsafe_allow_html=True)

    st.markdown('<div class="fs-label">Fraud Threshold</div>', unsafe_allow_html=True)
    threshold = st.slider(
        "threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        label_visibility="collapsed",
    )
    st.caption(f"Transactions above **{threshold:.0%}** probability are flagged as fraud.")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_single, tab_bulk = st.tabs(["Analyze Transaction", "Bulk Upload"])

with tab_single:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown('<div class="fs-label">Transaction</div>', unsafe_allow_html=True)
        Time = st.number_input("Time elapsed (seconds)", min_value=0.01, value=10000.0)
        Amount = st.number_input("Amount (₹)", min_value=0.01, value=100.0)

        with st.expander("PCA Components — V1 to V28"):
            st.caption("Anonymized features from PCA dimensionality reduction.")
            V = {}
            ca, cb = st.columns(2)
            for i in range(1, 29):
                with (ca if i % 2 != 0 else cb):
                    V[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.01, key=f"v{i}")

        analyze = st.button("Analyze", use_container_width=True)

    with right:
        if analyze:
            payload = {"Time": Time, **V, "Amount": Amount}
            try:
                with st.spinner(""):
                    resp = requests.post(PREDICT_URL, json=payload, timeout=15)

                if resp.status_code == 200:
                    data = resp.json()
                    prob = data.get("fraud_probability", 0)
                    pct = prob * 100

                    if prob >= threshold:
                        card, pc, vc = "card-high", "p-high", "v-high"
                        verdict = "Likely Fraudulent"
                    elif prob >= threshold * 0.6:
                        card, pc, vc = "card-mid", "p-mid", "v-mid"
                        verdict = "Moderate Risk"
                    else:
                        card, pc, vc = "card-low", "p-low", "v-low"
                        verdict = "Likely Legitimate"

                    st.session_state.result = {
                        "prob": prob, "pct": pct,
                        "card": card, "pc": pc, "vc": vc,
                        "verdict": verdict,
                        "amount": Amount, "time": Time, "threshold": threshold,
                    }
                    st.session_state.count += 1

                else:
                    st.error(f"API returned {resp.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API.")
            except Exception as e:
                st.error(str(e))

        res = st.session_state.result
        if res:
            label = "s" if st.session_state.count != 1 else ""
            st.markdown(f"""
            <div class="fs-card {res['card']}">
                <div class="fs-prob-unit">Fraud probability</div>
                <div class="fs-prob {res['pc']}">{res['pct']:.1f}%</div>
                <div class="fs-verdict {res['vc']}">{res['verdict']}</div>
                <div class="fs-meta">
                    threshold {res['threshold']:.0%}&ensp;·&ensp;₹{res['amount']:,.2f}&ensp;·&ensp;{res['time']:,.0f}s
                </div>
                <div class="fs-count">{st.session_state.count} transaction{label} analyzed this session</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(res["prob"], 1.0))
        else:
            st.markdown("""
            <div class="fs-card card-idle">
                <div class="fs-prob-unit">Fraud probability</div>
                <div class="fs-prob p-idle">—.—%</div>
                <div class="fs-verdict v-idle">No result yet</div>
                <div class="fs-meta">
                    Fill in the transaction fields on the left<br>and click Analyze.
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab_bulk:
    uploaded = st.file_uploader("Upload CSV — must match the API input schema", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(5), use_container_width=True)

        if st.button("Run Bulk Analysis"):
            results = []
            bar = st.progress(0)
            n = len(df)

            for i, (_, row) in enumerate(df.iterrows()):
                try:
                    r = requests.post(PREDICT_URL, json=row.to_dict(), timeout=15)
                    if r.status_code == 200:
                        res = r.json()
                        p = res.get("fraud_probability", 0)
                        res["risk"] = (
                            "High" if p >= threshold
                            else "Moderate" if p >= threshold * 0.6
                            else "Low"
                        )
                        results.append(res)
                    else:
                        results.append({"fraud_probability": None, "fraud_label": "Error", "risk": "Error"})
                except Exception:
                    results.append({"fraud_probability": None, "fraud_label": "Error", "risk": "Error"})
                bar.progress((i + 1) / n)

            results_df = pd.DataFrame(results)

            if "risk" in results_df.columns:
                c1, c2, c3 = st.columns(3)
                c1.metric("High Risk", len(results_df[results_df.risk == "High"]))
                c2.metric("Moderate", len(results_df[results_df.risk == "Moderate"]))
                c3.metric("Low Risk", len(results_df[results_df.risk == "Low"]))

            st.dataframe(results_df, use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode()
            st.download_button("Download results (.csv)", csv_bytes, "fraudshield_results.csv", "text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("FraudShield · Naman Gupta")
