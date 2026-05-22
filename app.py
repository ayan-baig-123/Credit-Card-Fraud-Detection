import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# =========================
# CUSTOM CSS (PREMIUM UI)
# =========================
st.markdown("""
<style>

/* ===== BACKGROUND ANIMATION ===== */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 10% 20%, #0f172a, #020617 40%, #000000);
    overflow: hidden;
}

/* moving neon glow orbs */
[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: absolute;
    width: 500px;
    height: 500px;
    border-radius: 50%;
    filter: blur(120px);
    opacity: 0.4;
    z-index: 0;
    animation: float 10s infinite alternate ease-in-out;
}

[data-testid="stAppViewContainer"]::before {
    background: #3b82f6;
    top: -100px;
    left: -100px;
}

[data-testid="stAppViewContainer"]::after {
    background: #22c55e;
    bottom: -150px;
    right: -120px;
    animation-delay: 2s;
}

@keyframes float {
    from { transform: translateY(0px) scale(1); }
    to { transform: translateY(60px) scale(1.2); }
}

/* ===== TITLE ===== */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: 900;
    color: #60a5fa;
    text-shadow: 0 0 15px #3b82f6, 0 0 40px #2563eb;
    margin-bottom: 30px;
}

/* glow animation */
@keyframes glow {
    0% { text-shadow: 0 0 10px #3b82f6; }
    50% { text-shadow: 0 0 25px #60a5fa, 0 0 50px #2563eb; }
    100% { text-shadow: 0 0 10px #3b82f6; }
}

.title {
    animation: glow 2.5s infinite ease-in-out;
}

/* ===== GLASS CARD ===== */
.block-container {
    padding: 2rem;
    z-index: 2;
    position: relative;
}

/* container cards */
div[data-testid="stVerticalBlock"] {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    transition: 0.3s ease;
}

div[data-testid="stVerticalBlock"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 25px rgba(59,130,246,0.3);
}

/* ===== INPUTS ===== */
input, textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: white !important;
    border-radius: 10px !important;
}

input:hover, textarea:hover {
    box-shadow: 0 0 15px rgba(59,130,246,0.4);
    transform: scale(1.02);
}

/* ===== BUTTON ===== */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #3b82f6, #22c55e);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 10px;
    border: none;
    transition: 0.3s;
    box-shadow: 0 0 10px rgba(59,130,246,0.4);
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px #3b82f6, 0 0 40px #22c55e;
}

/* ===== RESULT BOX ===== */
.result {
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 18px;
    font-weight: 600;
    margin-top: 20px;
    animation: pop 0.5s ease;
}

@keyframes pop {
    from { transform: scale(0.8); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
}

.safe {
    background: rgba(34,197,94,0.15);
    border: 1px solid #22c55e;
    color: #86efac;
    box-shadow: 0 0 15px rgba(34,197,94,0.3);
}

.fraud {
    background: rgba(239,68,68,0.15);
    border: 1px solid #ef4444;
    color: #fca5a5;
    box-shadow: 0 0 15px rgba(239,68,68,0.3);
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown("<div class='title'>💳 AI Fraud Detection System</div>", unsafe_allow_html=True)

st.write("")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data.csv")

# encoding
df["merchant_category"] = df["merchant_category"].astype("category").cat.codes
df = df.drop("transaction_id", axis=1)

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier()
model.fit(X_train, y_train)

# =========================
# UI LAYOUT
# =========================
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    amount = st.number_input("💰 Transaction Amount", 0.0)
    hour = st.slider("⏰ Transaction Hour", 0, 23)
    foreign = st.selectbox("🌍 Foreign Transaction", ["No", "Yes"])
    location = st.selectbox("📍 Location Mismatch", ["No", "Yes"])
    foreign = 1 if foreign == "Yes" else 0
    location = 1 if location == "Yes" else 0

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    trust = st.slider("🔐 Device Trust Score", 0, 100)
    velocity = st.number_input("⚡ Velocity (24h)", 0)
    age = st.number_input("👤 Cardholder Age", 18, 100)

    merchant = st.selectbox(
        "🏪 Merchant Category",
        ["Food", "Grocery", "Electronics", "Travel"]
    )

    st.markdown("</div>", unsafe_allow_html=True)

# encoding map
category_map = {
    "Food": 0,
    "Grocery": 1,
    "Electronics": 2,
    "Travel": 3
}

merchant_encoded = category_map[merchant]

# =========================
# PREDICTION BUTTON
# =========================
st.write("")
btn = st.button("🚀 Predict Fraud")

if btn:

    with st.spinner("Analyzing Transaction Pattern... 🔍"):

      input_data = pd.DataFrame(
    [[
        amount,
        hour,
        foreign,
        location,
        trust,
        velocity,
        age,
        merchant_encoded
    ]],
    columns=X.columns
    )

    prediction = model.predict(input_data)

    st.write("")

    # =========================
    # RESULT UI
    # =========================
    if prediction[0] == 1:
        st.markdown("""
        <div class='result-box fraud'>
        ⚠️ FRAUD DETECTED! Transaction is Suspicious.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='result-box safe'>
        ✅ LEGITIMATE TRANSACTION
        </div>
        """, unsafe_allow_html=True)

# =========================
# FOOTER ANIMATION
# =========================
st.write("")
st.markdown("---")
st.markdown("""
<div style="
    text-align:center;
    font-size:18px;
    font-weight:600;
    color:#60a5fa;
    text-shadow: 0 0 10px #3b82f6;
    margin-top:20px;
">
Built by Ayan Baig | Advanced AI Fraud Detection System | © 2026
</div>
""", unsafe_allow_html=True)
