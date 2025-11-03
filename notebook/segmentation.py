import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load models with full paths
kmeans = joblib.load(os.path.join(script_dir, 'kmeans_model.pkl'))
scaler = joblib.load(os.path.join(script_dir, 'scaler.pkl'))

st.title("Customer Segmentation")
st.write("Input customer data to predict their segment.")


Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
Total_Spend = st.number_input("Total Spend", min_value=0, max_value=5000, value=1000)
NumWebPurchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
NumStorePurchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=3)
NumWebVisitsMonth = st.number_input("Number of Web Visits per Month", min_value=0, max_value=50, value=3)
Recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

# Create input data with the EXACT same feature names and order as used in training
# Features: ['Age', 'Income', 'Total_Spend', 'Recency', 'NumWebPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
input_data = pd.DataFrame({
    'Age': [Age],
    'Income': [Income],
    'Total_Spend': [Total_Spend],
    'Recency': [Recency],
    'NumWebPurchases': [NumWebPurchases],
    'NumStorePurchases': [NumStorePurchases],
    'NumWebVisitsMonth': [NumWebVisitsMonth]
})

scaled_data = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(scaled_data)[0]
    st.success(f"The predicted customer segment is: Cluster {cluster}")
    st.write("Cluster Characteristics:")
    cluster_summary = pd.read_csv(os.path.join(script_dir, 'cluster_summary.csv'))
    cluster_info = cluster_summary[cluster_summary['Cluster'] == cluster]
    if not cluster_info.empty:
        st.write(cluster_info)
    else:
        st.write(f"Cluster {cluster} information not found in summary.")
