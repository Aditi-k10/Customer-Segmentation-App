import streamlit as st
import numpy as np
import joblib

# Load Model and Scaler
model = joblib.load("customer_segmentation_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Segmentation Prediction App")

st.write("Enter Customer Details")

# User Inputs
income = st.number_input("Income")
recency = st.number_input("Recency")
age = st.number_input("Age")
spending = st.number_input("Total Spending")
family_size = st.number_input("Family Size")
web_purchase = st.number_input("Number of Web Purchases")
catalog_purchase = st.number_input("Number of Catalog Purchases")
store_purchase = st.number_input("Number of Store Purchases")
web_visits = st.number_input("Web Visits Per Month")

if st.button("Predict Customer Segment"):

    input_data = np.array([[income, recency, age, spending, family_size,
                            web_purchase, catalog_purchase, store_purchase, web_visits]])

    scaled_data = scaler.transform(input_data)

    cluster = model.predict(scaled_data)

    # Convert cluster number into customer type
    if cluster[0] == 0:
        st.success("Low Value Customers")

    elif cluster[0] == 1:
        st.success("Premium High Spending Customers")

    elif cluster[0] == 2:
        st.success("Digital Active Customers")

    else:
        st.success("Family Oriented Moderate Customers")
