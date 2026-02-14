import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("customer_segmentation_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Title Section
# -----------------------------
st.title("📊 Customer Segmentation Prediction")
st.markdown("### Identify Customer Type Using Behaviour & Spending Patterns")

st.write("---")

# -----------------------------
# Sidebar Input Section
# -----------------------------
st.sidebar.header("Enter Customer Details")

income = st.sidebar.number_input("💰 Income", min_value=0)
recency = st.sidebar.number_input("🕒 Recency (Days since last purchase)", min_value=0)
age = st.sidebar.number_input("🎂 Age", min_value=0)
spending = st.sidebar.number_input("🛒 Total Spending", min_value=0)
family_size = st.sidebar.number_input("👨‍👩‍👧 Family Size", min_value=1)

web_purchase = st.sidebar.number_input("🌐 Web Purchases", min_value=0)
catalog_purchase = st.sidebar.number_input("📦 Catalog Purchases", min_value=0)
store_purchase = st.sidebar.number_input("🏬 Store Purchases", min_value=0)
web_visits = st.sidebar.number_input("💻 Monthly Web Visits", min_value=0)

# -----------------------------
# Prediction Button
# -----------------------------
if st.sidebar.button("🔍 Predict Customer Segment"):

    input_data = np.array([[income, recency, age, spending, family_size,
                            web_purchase, catalog_purchase,
                            store_purchase, web_visits]])

    scaled_data = scaler.transform(input_data)
    cluster = model.predict(scaled_data)[0]

    st.subheader("🎯 Prediction Result")

    # -----------------------------
    # Cluster Mapping
    # -----------------------------
    if cluster == 0:
        st.success("Low Value Customers")
        st.info("These customers spend less and are price sensitive.")

    elif cluster == 1:
        st.success("Premium High Spending Customers")
        st.info("High income and high spending customers. Loyal and valuable.")

    elif cluster == 2:
        st.success("Digital Active Customers")
        st.info("Customers who prefer online shopping and digital interaction.")

    else:
        st.success("Family Oriented Moderate Customers")
        st.info("Customers with larger families and moderate spending behaviour.")

# -----------------------------
# Footer
# -----------------------------
st.write("---")
st.markdown("Made with ❤️ using Streamlit")
