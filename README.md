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

---

# 🩺 Liver Disease Classification Using Machine Learning

## 🚀 Project Overview
This project focuses on predicting liver disease conditions using machine learning classification techniques.

The objective is to classify patients into different liver disease categories based on clinical and biochemical parameters obtained from blood and urine analysis.

Multiple machine learning models were developed and evaluated. The Gradient Boosting Classifier was selected as the final model due to its superior performance.

The final model is deployed as an interactive web application using Streamlit, allowing users to enter patient details and obtain real-time predictions.

---

## 🎯 Business Objective
This is a multi-class classification problem.

The goal is to accurately classify patients into one of the following categories:

- No Disease  
- Suspect Disease  
- Hepatitis C  
- Fibrosis  
- Cirrhosis  

Early and accurate classification supports timely medical diagnosis and intervention.

---

## 📊 Dataset Description
- Number of Instances: 615  
- Number of Variables: 13  
- Domain: Healthcare / Medical Data  
- Most input features are numerical  
- One binary feature: Sex  
- Data consists of laboratory measurements related to liver and kidney functions  

---

## 🎯 Target Variable
Category (Diagnosis):

- no_disease  
- suspect_disease  
- hepatitis_c  
- fibrosis  
- cirrhosis  

---

## 📌 Input Features Description

- Age (0–100 years)  
- Sex (Male / Female)  
- Albumin (34–54 g/L)  
- Alkaline Phosphatase (40–129 U/L)  
- Alanine Aminotransferase – ALT (7–55 U/L)  
- Aspartate Aminotransferase – AST (8–48 U/L)  
- Bilirubin (1–12 mg/L)  
- Cholinesterase (8–18 U/L)  
- Cholesterol (< 5.2 mmol/L)  
- Creatinina  
  - Male: 61.9–114.9 µmol/L  
  - Female: 53–97.2 µmol/L  
- Gamma Glutamyl Transferase – GGT (0–30/50 IU/L)  
- Protein (< 80 mg)  

---

## 🤖 Machine Learning Approach
- Formulated as a multi-class classification problem  
- Trained and evaluated multiple ML models  
- Dataset was imbalanced across disease categories  
- Selected Gradient Boosting Classifier due to:
  - Better handling of imbalanced data  
  - Strong performance on complex relationships  
  - Iterative error reduction mechanism  

Gradient Boosting improves performance by sequentially correcting errors made by previous models.

---

## 🖥 Streamlit Web Application
The deployed application provides:

- User-friendly data input interface  
- Real-time liver disease prediction  
- Probability scores for each disease category  
- Clear visual feedback of prediction results  

---

## 🌐 Deployment
The application is deployed using Streamlit Cloud.

Live Application:
https://liverdiseaseprediction-npkixarr2hxnu6earjedhj.streamlit.app/

---

## 👩‍💻 Author
Aditi Khande  
Final Year Computer Science Engineering Student  
Aspiring Data Scientist
