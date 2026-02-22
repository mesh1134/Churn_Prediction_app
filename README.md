# 📉 Telecom Customer Churn Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churnpredictionapp-afecolrw8tzptdbxsmanwk.streamlit.app/)

## 📌 Project Overview
This project is an end-to-end Machine Learning pipeline designed to predict customer churn in the telecommunications industry. By analyzing customer demographics, account information, and service usage, the model identifies high-risk customers, allowing businesses to take proactive retention measures.

## 🚀 Live Web Application
The predictive model is deployed as an interactive web application using Streamlit.
* **Try it live here:** https://churnpredictionapp-afecolrw8tzptdbxsmanwk.streamlit.app/

## 🧠 Technical Architecture
* **Data Preprocessing:** Implemented a robust Scikit-Learn `ColumnTransformer` pipeline to automatically handle scaling (StandardScaler) and One-Hot Encoding, preventing data leakage and training-serving skew.
* **Automated Model Selection:** Utilized a custom automated ML script (`mesh_utils_optimized`) utilizing `RandomizedSearchCV` to dynamically train, tune, and evaluate multiple algorithms (Random Forest, XGBoost, Logistic Regression, SVM).
* **Class Imbalance Handling:** Optimized the models specifically for **F1-Score** and utilized balanced class weights to heavily penalize the model for missing actual churners.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Scikit-Learn, XGBoost, Pandas, NumPy, Joblib
* **Deployment:** Streamlit Community Cloud

## 💻 How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `python -m streamlit run churn_app.py`
