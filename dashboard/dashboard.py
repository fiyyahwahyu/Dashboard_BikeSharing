import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set up the Streamlit app
st.set_page_config(page_title="Sharing Sepeda", layout="wide")
st.title("📊 Sharing Sepeda: Analisis Data")

# Load data
@st.cache_data
def load_data():
    df_hour = pd.read_csv("https://drive.google.com/uc?id=1L9OyIKdgGVPpThq55Z-54BFpOfurkNVu&export=download")
    df_day = pd.read_csv("https://drive.google.com/uc?id=1LloLN-1f6CXvF4Qpp7Wgy4a6gO9KZtgu&export=download")
    return df_hour, df_day

df_hour, df_day = load_data()

# Sidebar for navigation
st.sidebar.title("Navigasi")
option = st.sidebar.radio("Pilih Halaman:", ("Analisis Jam", "Analisis Hari"))

# Data Analysis Section for Hourly Data
if option == "Analisis Jam":
    st.subheader("Analisis Data Jam")
    
    if st.checkbox("Tampilkan Data Jam"):
        st.write(df_hour.head())

    st.write("Jumlah missing values di df_hour:", df_hour.isna().sum().sum())
    st.write("Jumlah duplikat di df_hour:", df_hour.duplicated().sum())

    # Replace season numbers with names
    season_mapping = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    df_hour['season'] = df_hour['season'].replace(season_mapping)

    # Group by season
    hour_season_counts = df_hour.groupby(by="season")['cnt'].sum().sort_index()

    # Bar plot for season counts
    plt.figure(figsize=(10, 6))
    plt.bar(hour_season_counts.index, hour_season_counts.values, color='skyblue')
    plt.title('Total Sharing Sepeda Berdasarkan Musim (Season)', fontsize=18)
    plt.xlabel('Musim', fontsize=14)
    plt.ylabel('Total Sharing', fontsize=14)
    plt.xticks(hour_season_counts.index)
    st.pyplot(plt)

    # Monthly analysis
    hour_monthly_cnt = df_hour.groupby(by="mnth")[['casual', 'registered', 'cnt']].sum().sort_index()

    # Bar plot for monthly counts
    plt.figure(figsize=(10, 6))
    months = hour_monthly_cnt.index
    bar_width = 0.35
    index = np.arange(len(months))
    plt.bar(index - bar_width/2, hour_monthly_cnt['cnt'], bar_width, label='Total', color='skyblue')
    plt.title('Total Sharing Sepeda Berdasarkan Bulan', fontsize=18)
    plt.xlabel('Bulan', fontsize=14)
    plt.ylabel('Total Sharing', fontsize=14)
    plt.xticks(index, months)
    plt.legend()
    st.pyplot(plt)

# Data Analysis Section for Daily Data
if option == "Analisis Hari":
    st.subheader("Analisis Data Hari")

    if st.checkbox("Tampilkan Data Hari"):
        st.write(df_day.head())

    df_day['season'] = df_day['season'].replace(season_mapping)
    day_season_counts = df_day.groupby(by="season")['cnt'].sum().sort_index()

    # Bar plot for daily season counts
    plt.figure(figsize=(10, 6))
    plt.bar(day_season_counts.index, day_season_counts.values, color='skyblue')
    plt.title('Total Sharing Sepeda Berdasarkan Musim (Season)', fontsize=18)
    plt.xlabel('Musim', fontsize=14)
    plt.ylabel('Total Sharing', fontsize=14)
    plt.xticks(day_season_counts.index)
    st.pyplot(plt)

    # Monthly analysis for daily data
    day_monthly_cnt = df_day.groupby(by="mnth")[['casual', 'registered', 'cnt']].sum().sort_index()

    # Bar plot for monthly counts
    plt.figure(figsize=(10, 6))
    plt.bar(index - bar_width/2, day_monthly_cnt['cnt'], bar_width, label='Total', color='skyblue')
    plt.title('Total Sharing Sepeda Berdasarkan Bulan', fontsize=18)
    plt.xlabel('Bulan', fontsize=14)
    plt.ylabel('Total Sharing', fontsize=14)
    plt.xticks(index, months)
    plt.legend()
    st.pyplot(plt)

# Clustering analysis example
st.subheader("Clustering Sharing")
features = df_hour[['casual', 'registered']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Manually set centroids for illustration purposes
centroid_1 = features_scaled[0]
centroid_2 = features_scaled[1]

def euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

clusters = []
for point in features_scaled:
    dist_to_centroid_1 = euclidean_distance(point, centroid_1)
    dist_to_centroid_2 = euclidean_distance(point, centroid_2)
    clusters.append(0 if dist_to_centroid_1 < dist_to_centroid_2 else 1)

df_hour['cluster'] = clusters

# Scatter plot for clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_hour, x='casual', y='registered', hue='cluster', palette='viridis', alpha=0.7)
plt.title('Clustering Sharing Sepeda', fontsize=18)
plt.xlabel('Jumlah Casual', fontsize=14)
plt.ylabel('Jumlah Registered', fontsize=14)
plt.legend(title='Cluster')
st.pyplot(plt)

# Footer
st.markdown("---")
st.markdown("Data diperoleh dari dataset Sharing sepeda. Analisis ini bertujuan untuk memahami pola penggunaan sepeda dan meningkatkan layanan.")
