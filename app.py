import streamlit as st
import numpy as np
import joblib

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>

/* Remove Streamlit top header + white rounded bar */
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
div[data-testid="stToolbar"] {display: none;}
div[data-testid="stDecoration"] {display: none;}

/* App background */
.stApp {
    background-color: #f4f8fc;
}

/* Remove extra top spacing */
.block-container {
    padding-top: 0rem;
}

/* ===== Main Title ===== */
.app-title {
    font-size:60px;
    font-weight:800;
    text-align:center;
    color:#2F80ED;
    margin-top:20px;
    margin-bottom:10px;
}

/* ===== Subtitle ===== */
.app-subtitle {
    text-align:center;
    font-size:20px;
    color:#5f6c7b;
    margin-bottom:40px;
}

/* ===== Sidebar Styling (Light Modern) ===== */
section[data-testid="stSidebar"] {
    background-color: #EAF2FD;
    width: 400px !important;
    padding: 25px;
}

section[data-testid="stSidebar"] label {
    color: #2F80ED !important;
    font-weight:600;
}

section[data-testid="stSidebar"] .stNumberInput input {
    border-radius: 10px;
}

/* Sidebar Button */
section[data-testid="stSidebar"] button {
    background-color: #2F80ED !important;
    color: white !important;
    font-weight:600;
    border-radius: 12px;
}

/* ===== Cards ===== */
.card {
    background: white;
    padding: 35px;
    border-radius: 18px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.05);
    margin-bottom: 30px;
}

/* Section Titles */
h3 {
    font-size:30px !important;
    font-weight:700 !important;
    color:#2F80ED !important;
}

/* Content Text */
.card p, .card div {
    font-size:19px !important;
    line-height:1.7;
}

/* Result Box */
.result-box {
    padding: 35px;
    border-radius: 18px;
    background-color: #ffffff;
    text-align:center;
    font-size:26px;
    font-weight:700;
    border: 2px solid #2F80ED;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
scaler = joblib.load("scaler.pkl")
model = joblib.load("customer_segmentation_model.pkl")

# -------------------- Title --------------------
st.markdown('<div class="app-title">Customer Segmentation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Predict Customer Type Using Behaviour & Spending Pattern</div>', unsafe_allow_html=True)

# -------------------- Sidebar --------------------
st.sidebar.markdown("## 📋 Enter Customer Details")

income = st.sidebar.number_input("💰 Income", min_value=0.0)
recency = st.sidebar.number_input("⏳ Recency (Days Since Last Purchase)", min_value=0)
age = st.sidebar.number_input("🎂 Age", min_value=0)
total_spending = st.sidebar.number_input("🛒 Total Spending", min_value=0.0)
family_size = st.sidebar.number_input("👨‍👩‍👧 Family Size", min_value=1)

num_web_purchases = st.sidebar.number_input("🌐 Web Purchases", min_value=0)
num_catalog_purchases = st.sidebar.number_input("📦 Catalog Purchases", min_value=0)
num_store_purchases = st.sidebar.number_input("🏬 Store Purchases", min_value=0)
num_web_visits = st.sidebar.number_input("💻 Web Visits Per Month", min_value=0)

predict_btn = st.sidebar.button("🚀 Predict Segment")

# -------------------- Customer Profile --------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Customer Profile")

st.write(f"""
**Income:** {income}  
**Age:** {age}  
**Total Spending:** {total_spending}  
**Recency:** {recency} days  
**Family Size:** {family_size}
""")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Purchase Behaviour --------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 Purchase Behaviour")

st.write(f"""
**Web Purchases:** {num_web_purchases}  
**Catalog Purchases:** {num_catalog_purchases}  
**Store Purchases:** {num_store_purchases}  
**Web Visits Per Month:** {num_web_visits}
""")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Prediction --------------------
if predict_btn:

    input_data = np.array([[income, recency, age, total_spending,
                            family_size, num_web_purchases,
                            num_catalog_purchases, num_store_purchases,
                            num_web_visits]])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    st.subheader("🎯 Prediction Result")

    if prediction == 0:
        result_text = "Cluster 0 → 💡 Low Income Browsing Customers"
    elif prediction == 1:
        result_text = "Cluster 1 → 👑 Affluent Premium Customers"
    elif prediction == 2:
        result_text = "Cluster 2 → ⭐ Active Customers"
    else:
        result_text = "Cluster 3 → 👨‍👩‍👧 Family Oriented Moderate Customers"

    st.markdown(f'<div class="result-box">{result_text}</div>', unsafe_allow_html=True)
    st.success("Prediction Completed Successfully ✅")

# -------------------- Footer --------------------
st.markdown("---")
