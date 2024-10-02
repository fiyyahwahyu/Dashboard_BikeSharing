import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set up the page configuration
st.set_page_config(
    page_title="Bike Rentals Analysis Dashboard",
    page_icon="🚲",
    layout="wide",  # Use 'wide' layout for a more spacious dashboard
    initial_sidebar_state="expanded"
)

# Set up custom CSS styling for more appealing design (optional)
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    h1 {
        color: #4CAF50;
    }
    .sidebar .sidebar-content {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Title of the dashboard
st.title("🚲 Bike Rentals Analysis and Clustering Dashboard")

# Load data from URLs
@st.cache_data
def load_data():
    df_hour = pd.read_csv("https://drive.google.com/uc?id=1L9OyIKdgGVPpThq55Z-54BFpOfurkNVu&export=download")
    df_day = pd.read_csv("https://drive.google.com/uc?id=1LloLN-1f6CXvF4Qpp7Wgy4a6gO9KZtgu&export=download")
    return df_hour, df_day

df_hour, df_day = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("Use the sections below to explore data and clustering.")

# Show dataframes
st.sidebar.subheader("View Data")
if st.sidebar.checkbox("Show Hourly Data"):
    st.subheader("Hourly Data")
    st.dataframe(df_hour.head())

if st.sidebar.checkbox("Show Daily Data"):
    st.subheader("Daily Data")
    st.dataframe(df_day.head())

# Show data info, missing values, and duplicates
st.sidebar.subheader("Data Insights")
if st.sidebar.checkbox("Show Data Overview"):
    st.subheader("Data Overview")
    st.write("Hourly Data Info:")
    st.write(df_hour.info())

    st.write("Missing Values in Hourly Data:")
    st.write(df_hour.isna().sum())

    st.write("Duplicates in Hourly Data:")
    st.write(df_hour.duplicated().sum())

    st.write("Daily Data Info:")
    st.write(df_day.info())

    st.write("Missing Values in Daily Data:")
    st.write(df_day.isna().sum())

    st.write("Duplicates in Daily Data:")
    st.write(df_day.duplicated().sum())

# Aggregating hourly data by day
df_hour_agg = df_hour.groupby('dteday').agg({
    'casual': 'sum',
    'registered': 'sum',
    'cnt': 'sum'
}).reset_index()

# Merging day and hour data
df_combined = pd.merge(df_day, df_hour_agg, on='dteday', suffixes=('_day', '_hour'))

# Displaying the combined data (for debugging purposes)
st.write("### Combined Data Sample:")
st.write(df_combined.head())

# Total Sharings by season and month
total_by_season_mnth = df_combined.groupby(['season', 'mnth']).agg({
    'cnt_day': 'sum',
    'cnt_hour': 'sum'
}).reset_index()

# Displaying total Sharings by season and month
st.write("### Total Sharings by Season and Month:")
st.write(total_by_season_mnth)

# Total Daily Sharings by Season and Month
st.subheader('Total Daily Sharings Based on Season and Month')
fig1, ax1 = plt.subplots(figsize=(12, 8))
sns.barplot(data=total_by_season_mnth, x='mnth', y='cnt_day', hue='season', ax=ax1)
ax1.set_title('Total Daily Bike Sharings Based on Season and Month')
ax1.set_xlabel('Month')
ax1.set_ylabel('Total Sharings')
plt.xticks(rotation=45)
st.pyplot(fig1)

# Total Hourly Sharings by Season and Month
st.subheader('Total Hourly Sharings Based on Season and Month')
fig2, ax2 = plt.subplots(figsize=(12, 8))
sns.barplot(data=total_by_season_mnth, x='mnth', y='cnt_hour', hue='season', ax=ax2)
ax2.set_title('Total Hourly Bike Sharings Based on Season and Month')
ax2.set_xlabel('Month')
ax2.set_ylabel('Total Sharings')
plt.xticks(rotation=45)
st.pyplot(fig2)

# Total Sharings by Season and Year
total_by_season_year = df_combined.groupby(['yr', 'season']).agg({
    'cnt_day': 'sum',
    'cnt_hour': 'sum'
}).reset_index()
total_by_season_year['yr'] = total_by_season_year['yr'].replace({0: 2011, 1: 2012})

# Visualization for Total Sharings by Season and Year
st.subheader('Total Daily Sharings by Season for Years 2011 and 2012')
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.barplot(data=total_by_season_year, x='season', y='cnt_day', hue='yr', ax=ax3)
ax3.set_title('Total Daily Bike Sharings by Season (2011 vs 2012)')
ax3.set_xlabel('Season')
ax3.set_ylabel('Total Sharings (Days)')
ax3.legend(title='Year', loc='upper right', labels=['2011', '2012'])
st.pyplot(fig3)

# Visualization for season-based rentals
st.sidebar.subheader("Visualizations")
if st.sidebar.checkbox("Show Seasonal Rentals Visualization"):
    st.subheader("Seasonal Rentals Visualization")

    season_mapping = {
        1: 'Spring',
        2: 'Summer',
        3: 'Fall',
        4: 'Winter'
    }
    df_hour['season'] = df_hour['season'].replace(season_mapping)

    hour_season_counts = df_hour.groupby(by="season")['cnt'].sum().sort_index()

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(hour_season_counts.index, hour_season_counts.values, color='skyblue')
    ax1.set_title('Total daily tenant based on season', fontsize=16)
    ax1.set_xlabel('Season', fontsize=14)
    ax1.set_ylabel('Total Tenant', fontsize=14)
    st.pyplot(fig1)

# Visualization for month-based rentals
if st.sidebar.checkbox("Show Monthly Rentals Visualization"):
    st.subheader("Monthly Rentals Visualization")

    hour_monthly_cnt = df_hour.groupby(by="mnth")[['cnt']].sum().sort_index()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    months = hour_monthly_cnt.index
    bar_width = 0.35
    index = np.arange(len(months))
    ax2.bar(index, hour_monthly_cnt['cnt'], bar_width, color='skyblue')
    ax2.set_title('Total Hourly Tenant Based on Month', fontsize=16)
    ax2.set_xlabel('Month', fontsize=14)
    ax2.set_ylabel('Tenant (cnt)', fontsize=14)
    ax2.set_xticks(index)
    ax2.set_xticklabels(months)
    st.pyplot(fig2)

# Clustering Section with KMeans
if st.sidebar.checkbox("Show Clustering"):
    st.subheader("Clustering")

    # Data preparation
    df_hour['dteday'] = pd.to_datetime(df_hour['dteday'])
    df_hour['month'] = df_hour['dteday'].dt.month

    monthly_rentals = df_hour.groupby('month')['cnt'].sum().reset_index()
    total_hourly_rentals = df_hour.groupby('month')['cnt'].sum().reset_index(name='total_hour_rentals')
    combined_data = pd.merge(monthly_rentals, total_hourly_rentals, on='month')

    X = combined_data[['cnt', 'total_hour_rentals']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # User input for number of clusters
    n_clusters = st.slider("Select the number of clusters", 2, 10, 3)

    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    combined_data['cluster'] = kmeans.fit_predict(X_scaled)

    # Visualize Clusters
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    scatter = ax3.scatter(combined_data['cnt'], combined_data['total_hour_rentals'], c=combined_data['cluster'], cmap='viridis')
    ax3.set_title('Clustering Penyewa Sepeda', fontsize=16)
    ax3.set_xlabel('Jumlah Penyewaan per Bulan', fontsize=14)
    ax3.set_ylabel('Total Jam Penyewaan per Bulan', fontsize=14)
    plt.colorbar(scatter)
    st.pyplot(fig3)

    # Show the clusters' centers and inertia plot
    st.write("Cluster Centers:")
    st.write(kmeans.cluster_centers_)

    # Elbow method plot
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(K, inertia, marker='o')
    ax4.set_title('Metode Elbow untuk Menentukan Jumlah Cluster', fontsize=16)
    ax4.set_xlabel('Jumlah Cluster', fontsize=14)
    ax4.set_ylabel('Inertia', fontsize=14)
    st.pyplot(fig4)