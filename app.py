import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- REMOVE TOP BAR COMPLETELY ----------------
st.markdown("""
<style>

/* Hide Streamlit header, toolbar, footer */
header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
div[data-testid="stToolbar"] {display: none;}
div[data-testid="stDecoration"] {display: none;}

/* Remove top padding */
.block-container {
    padding-top: 0rem;
}

/* App background */
.stApp {
    background-color: #f5f5f5;
}

/* Sidebar simple light */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e0e0e0;
}

/* Card style */
.card {
    background: white;
    padding: 25px;
    border-radius: 10px;
    margin-bottom: 20px;
    border: 1px solid #e5e5e5;
}

/* Result box */
.result-box {
    padding: 25px;
    border-radius: 10px;
    background: white;
    border: 1px solid #333;
    font-size: 20px;
    font-weight: 600;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
scaler = joblib.load("scaler.pkl")
model = joblib.load("customer_segmentation_model.pkl")

# ---------------- TITLE ----------------
st.title("Customer Segmentation Dashboard")
st.write("Predict customer segment using spending behaviour.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Enter Customer Details")

income = st.sidebar.number_input("Income", min_value=0.0)
recency = st.sidebar.number_input("Recency (Days)", min_value=0)
age = st.sidebar.number_input("Age", min_value=0)
total_spending = st.sidebar.number_input("Total Spending", min_value=0.0)
family_size = st.sidebar.number_input("Family Size", min_value=1)

num_web_purchases = st.sidebar.number_input("Web Purchases", min_value=0)
num_catalog_purchases = st.sidebar.number_input("Catalog Purchases", min_value=0)
num_store_purchases = st.sidebar.number_input("Store Purchases", min_value=0)
num_web_visits = st.sidebar.number_input("Web Visits Per Month", min_value=0)

predict_btn = st.sidebar.button("Predict")

# ---------------- PROFILE CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Customer Profile")

st.write(f"""
Income: {income}  
Age: {age}  
Total Spending: {total_spending}  
Recency: {recency}  
Family Size: {family_size}
""")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PURCHASE CARD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Purchase Behaviour")

st.write(f"""
Web Purchases: {num_web_purchases}  
Catalog Purchases: {num_catalog_purchases}  
Store Purchases: {num_store_purchases}  
Web Visits: {num_web_visits}
""")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if predict_btn:

    input_data = np.array([[income, recency, age, total_spending,
                            family_size, num_web_purchases,
                            num_catalog_purchases, num_store_purchases,
                            num_web_visits]])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    if prediction == 0:
        result_text = "Cluster 0 - Low Income Customers"
    elif prediction == 1:
        result_text = "Cluster 1 - Premium Customers"
    elif prediction == 2:
        result_text = "Cluster 2 - Active Customers"
    else:
        result_text = "Cluster 3 - Family Customers"

    st.markdown(f'<div class="result-box">{result_text}</div>', unsafe_allow_html=True)
