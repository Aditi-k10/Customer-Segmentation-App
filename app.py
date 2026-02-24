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

/* Increase Sidebar Width */
section[data-testid="stSidebar"] {
    width: 450px !important;
}
section[data-testid="stSidebar"] > div {
    width: 450px !important;
}

/* Main Title */
.main-title {
    font-size:55px;
    font-weight:800;
    color:#1B4F72;
    margin-top:10px;
    margin-bottom:0px;
}

/* Subtitle */
.sub-text {
    font-size:20px;
    color:#5D6D7E;
    margin-bottom:30px;
}

/* Section Boxes */
.section-box {
    padding:25px;
    border-radius:18px;
    background-color:#F4F6F7;
    box-shadow:0px 4px 12px rgba(0,0,0,0.05);
    margin-bottom:25px;
}

/* Prediction Box */
.result-box {
    padding:30px;
    border-radius:18px;
    background-color:#EBF5FB;
    text-align:center;
    font-size:24px;
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

st.write("")

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

# -------------------- Layout --------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("📊 Customer Profile")
    st.write(f"""
    **Income:** {income}  
    **Age:** {age}  
    **Total Spending:** {total_spending}  
    **Recency:** {recency} days  
    **Family Size:** {family_size}
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
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

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("🎯 Prediction Result")

    if prediction == 0:
        st.markdown('<div class="result-box">Cluster 0 → 💡 Budget / Low Value Customers</div>', unsafe_allow_html=True)

    elif prediction == 1:
        st.markdown('<div class="result-box">Cluster 1 → 👑 Premium High Spending Customers</div>', unsafe_allow_html=True)

    elif prediction == 2:
        st.markdown('<div class="result-box">Cluster 2 → ⭐ Digital Active Customers</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="result-box">Cluster 3 → 👨‍👩‍👧 Family Oriented Moderate Customers</div>', unsafe_allow_html=True)

    st.success("Prediction Completed Successfully ✅")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Footer --------------------
st.markdown("---")
