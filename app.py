import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classifier Evaluation", layout="wide")
st.title("Machine Learning Classification Evaluation Dashboard")

# 1. Dataset Upload Option (CSV)
st.sidebar.header("1. Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Test Dataset Preview", data.head())
    
    default_target_idx = len(data.columns) - 1 if "target" not in data.columns else data.columns.get_loc("target")
    target_col = st.sidebar.selectbox("Select Target Column", data.columns, index=default_target_idx)
    
    X_test = data.drop(columns=[target_col])
    y_test = data[target_col]
    
    # 2. Model Selection Dropdown
    st.sidebar.header("2. Model Selection")
    model_choice = st.sidebar.selectbox(
        "Select Classification Model",
        ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
    )
    
    model_key = model_choice.lower().replace(" ", "_")
    model_path = f"model/{model_key}.pkl"
    scaler_path = "model/scaler.pkl"
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        if model_key in ["logistic_regression", "knn", "support_vector_machine"]:
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test.values
            
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # 3. Display Evaluation Metrics
        st.subheader(f"Evaluation Metrics: {model_choice}")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        col2.metric("AUC Score", f"{roc_auc_score(y_test, y_proba):.4f}")
        col3.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
        col4.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
        col5.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
        col6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")
        
        st.divider()
        
        # 4. Display Confusion Matrix and Classification Report
        col_cm, col_cr = st.columns(2)
        
        with col_cm:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            
        with col_cr:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred)
            st.code(report, language="text")
            
    except Exception as e:
        st.error(f"Error loading model or evaluating data: {e}")
else:
    st.info("Please upload `test_data.csv` using the sidebar to display model performance metrics.")
