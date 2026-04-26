import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("Customer Segmentation using K-Means")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Drop duplicates
    df = df.drop_duplicates()

    # One-hot encoding for region
    if 'region' in df.columns:
        df = pd.get_dummies(df, columns=['region'], drop_first=True)

    # Select features
    features = df[['annual_income', 'purchase_amount',
                   'purchase_frequency', 'loyalty_score']]

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    # Elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    st.subheader("Elbow Method Graph")
    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, 11), wcss, marker='o')
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("WCSS")
    st.pyplot(fig1)

    # Apply KMeans (fixed k=4)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    df['Cluster'] = clusters

    st.subheader("Clustered Data")
    st.write(df.head())

    # Visualization
    st.subheader("Customer Segmentation Plot")
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(df['annual_income'], df['purchase_amount'],
                          c=df['Cluster'])
    ax2.set_xlabel("Annual Income")
    ax2.set_ylabel("Purchase Amount")
    st.pyplot(fig2)

    # Cluster summary
    st.subheader("Cluster Summary")
    st.write(df.groupby('Cluster').mean())

else:
    st.info("Please upload a CSV file to proceed.")
