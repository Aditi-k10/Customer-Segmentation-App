import streamlit as st
import numpy as np
import joblib

# -------------------- Page Config --------------------
st.markdown("""
<style>

/* App background */
.stApp {
    background: linear-gradient(135deg, #eef2f7, #d9e4f5);
}

/* Remove extra top white spacing */
.block-container {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
}

/* Remove empty white bar */
header[data-testid="stHeader"] {
    background: transparent;
}

div[data-testid="stToolbar"] {
    visibility: hidden;
    height: 0%;
    position: fixed;
}

/* Main Title */
.app-title {
    font-size:65px;
    font-weight:900;
    text-align:center;
    color:#0B3C5D;
    margin-top:10px;
    margin-bottom:5px;
}

/* Subtitle */
.app-subtitle {
    text-align:center;
    font-size:20px;
    color:#34495E;
    margin-bottom:40px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B3C5D, #1D6FA3);
    width: 420px !important;   /* Increased width */
    padding: 30px;
}

/* Sidebar text color */
section[data-testid="stSidebar"] label {
    color: white !important;
    font-weight: 600;
}

section[data-testid="stSidebar"] .stNumberInput input {
    background-color: #f0f4f8;
}

/* Card style */
.card {
    background: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 30px;
}

/* Result box */
.result-box {
    padding: 35px;
    border-radius: 20px;
    background: #ffffff;
    text-align:center;
    font-size:26px;
    font-weight:700;
    border: 3px solid #0B3C5D;
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

# -------------------- Purchase Behaviour BELOW --------------------
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
        result_text = "Cluster 0 → 💡  Low Income Browsing Customers"
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

