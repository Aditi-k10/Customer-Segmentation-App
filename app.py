import streamlit as st
import numpy as np
import joblib

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍",
    layout="wide"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>

/* -------- Sidebar Width -------- */
section[data-testid="stSidebar"] {
    width: 450px !important;
}
section[data-testid="stSidebar"] > div {
    width: 450px !important;
}

/* -------- Add More Top Space -------- */
.block-container {
    padding-top: 3rem;
}

/* -------- Centered Main Title -------- */
.main-title {
    font-size:60px;
    font-weight:800;
    text-align:center;
    margin-bottom:10px;
    color:#1F4E79;
}

/* -------- Subtitle -------- */
.sub-text {
    font-size:22px;
    text-align:center;
    color:#6c757d;
    margin-bottom:50px;
}

/* -------- Section Box -------- */
.custom-box {
    background-color:#f8f9fa;
    padding:35px;
    border-radius:18px;
    margin-bottom:35px;
    box-shadow:0px 6px 18px rgba(0,0,0,0.06);
}

/* -------- Section Titles -------- */
.section-title {
    font-size:32px;
    font-weight:800;
    margin-bottom:20px;
    color:#2E86C1;
}

/* -------- Details Text -------- */
.details-text {
    font-size:20px;
    line-height:1.8;
    font-weight:500;
}

/* -------- Prediction Box -------- */
.result-box {
    padding:35px;
    border-radius:18px;
    background-color:#e9f2ff;
    text-align:center;
    font-size:28px;
    font-weight:700;
    border:2px solid #2E86C1;
}

</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
scaler = joblib.load("scaler.pkl")
model = joblib.load("customer_segmentation_model.pkl")

# -------------------- Title --------------------
st.markdown('<div class="main-title">🛍 Customer Segmentation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Predict Customer Type Using Behaviour & Spending Pattern</div>', unsafe_allow_html=True)

# -------------------- Sidebar Inputs --------------------
st.sidebar.header("📋 Enter Customer Details")

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
st.markdown('<div class="custom-box">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Customer Profile</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="details-text">
<b>Income:</b> {income}<br>
<b>Age:</b> {age}<br>
<b>Total Spending:</b> {total_spending}<br>
<b>Recency:</b> {recency} days<br>
<b>Family Size:</b> {family_size}
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Purchase Behaviour --------------------
st.markdown('<div class="custom-box">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📈 Purchase Behaviour</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="details-text">
<b>Web Purchases:</b> {num_web_purchases}<br>
<b>Catalog Purchases:</b> {num_catalog_purchases}<br>
<b>Store Purchases:</b> {num_store_purchases}<br>
<b>Web Visits Per Month:</b> {num_web_visits}
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Prediction --------------------
if predict_btn:

    input_data = np.array([[income, recency, age, total_spending,
                            family_size, num_web_purchases,
                            num_catalog_purchases, num_store_purchases,
                            num_web_visits]])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    st.markdown('<div class="custom-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)

    if prediction == 0:
        result_text = "Cluster 0 → 💡 Budget / Low Value Customers"
    elif prediction == 1:
        result_text = "Cluster 1 → 👑 Premium High Spending Customers"
    elif prediction == 2:
        result_text = "Cluster 2 → ⭐ Digital Active Customers"
    else:
        result_text = "Cluster 3 → 👨‍👩‍👧 Family Oriented Moderate Customers"

    st.markdown(f'<div class="result-box">{result_text}</div>', unsafe_allow_html=True)

    st.success("Prediction Completed Successfully ✅")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Footer --------------------
st.markdown("---")
