import streamlit as st
import numpy as np
import joblib

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="🛍",
    layout="wide"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>

.block-container {
    max-width: 900px;
    padding-top: 2rem;
}

.main-title {
    font-size:52px;
    font-weight:800;
    text-align:center;
    color:#1F4E79;
    margin-bottom:10px;
}

.sub-text {
    text-align:center;
    font-size:18px;
    color:#6c757d;
    margin-bottom:40px;
}

.card {
    padding:25px;
    border-radius:15px;
    background-color:#f8f9fa;
    margin-bottom:25px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
}

.result-box {
    padding:30px;
    border-radius:15px;
    background-color:#eef5ff;
    text-align:center;
    font-size:24px;
    font-weight:600;
    border:2px solid #1F4E79;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
scaler = joblib.load("scaler.pkl")
model = joblib.load("customer_segmentation_model.pkl")

# -------------------- Title Section --------------------
st.markdown('<p class="main-title">Customer Segmentation App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">AI-Powered Customer Profiling & Cluster Prediction</p>', unsafe_allow_html=True)

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

# -------------------- Customer Profile Card --------------------
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

# -------------------- Purchase Behaviour Card --------------------
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
        result_text = "Cluster 0 → 💡 Budget / Low Value Customers"
    elif prediction == 1:
        result_text = "Cluster 1 → 👑 Premium High Spending Customers"
    elif prediction == 2:
        result_text = "Cluster 2 → ⭐ Digital Active Customers"
    else:
        result_text = "Cluster 3 → 👨‍👩‍👧 Family Oriented Moderate Customers"

    st.markdown(f'<div class="result-box">{result_text}</div>', unsafe_allow_html=True)
    st.success("Prediction Completed Successfully ✅")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Developed by Aditi Khande | Data Science Project")
