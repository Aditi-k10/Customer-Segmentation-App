# 📊 Data-Driven Customer Profiling and Segmentation Framework

## 🚀 Project Overview
This project presents an end-to-end unsupervised machine learning framework for customer profiling and segmentation.

The objective is to analyze customer demographic and purchasing behavior to identify meaningful customer groups and generate actionable marketing insights.

The final clustering model is integrated into an interactive Streamlit web application that enables real-time customer segmentation predictions.

---

## 🎯 Business Objective
Businesses require structured customer segmentation to:

- Personalize marketing campaigns  
- Identify high-value customers  
- Improve customer retention  
- Optimize marketing budget allocation  

This project groups customers based on demographic and behavioral features using unsupervised learning techniques.

---

## 🧠 Methodology

### 1. Data Preprocessing
- Handled missing values  
- Performed feature engineering (Family Size creation)  
- Applied feature scaling  
- Managed outliers to improve clustering stability  

### 2. Model Development
- Applied K-Means Clustering (Unsupervised Learning)  
- Determined optimal number of clusters using:
  - Elbow Method  
  - Silhouette Score  

### 3. Cluster Interpretation
Identified four meaningful customer segments:
- Premium Customers  
- Budget Customers  
- Digital Active Customers  
- Family-Oriented Customers  

Derived data-driven marketing strategies for each segment.

---

## 🛠 Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  
- Git & GitHub  

---

## 🌐 Deployment
The trained clustering model is deployed using Streamlit Cloud, enabling users to input customer details and receive real-time segmentation results.

---
