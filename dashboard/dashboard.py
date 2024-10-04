import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Title of the app
st.title('Bike Rental Analysis')

# Data Gathering
df_hour = pd.read_csv("https://drive.google.com/uc?id=1L9OyIKdgGVPpThq55Z-54BFpOfurkNVu&export=download")
df_day = pd.read_csv("https://drive.google.com/uc?id=1LloLN-1f6CXvF4Qpp7Wgy4a6gO9KZtgu&export=download")

# Display the first few rows of the datasets
st.subheader("Hourly Data Sample")
st.dataframe(df_hour.head())
st.subheader("Daily Data Sample")
st.dataframe(df_day.head())

# Data Assessment
st.subheader("Missing Values in Hourly Dataset")
st.text(df_hour.isna().sum())
st.subheader("Unique Values in Hourly Dataset")
st.text(df_hour.nunique())
st.subheader("Duplicate Entries in Hourly Dataset")
st.text(df_hour.duplicated().sum())

st.subheader("Missing Values in Daily Dataset")
st.text(df_day.isna().sum())
st.subheader("Unique Values in Daily Dataset")
st.text(df_day.nunique())
st.subheader("Duplicate Entries in Daily Dataset")
st.text(df_day.duplicated().sum())

# Data Cleaning Summary
st.subheader("Cleaning Summary")
st.write("""
Both datasets are free of duplicates and missing values.
""")

# EDA: Hourly Data
st.subheader("Explore Hourly Data")

# Change season values
season_mapping = {
    1: 'Spring',
    2: 'Summer',
    3: 'Fall',
    4: 'Winter'
}
df_hour['season'] = df_hour['season'].replace(season_mapping)

# Total daily tenant based on season
hour_season_counts = df_hour.groupby(by="season")['cnt'].sum().sort_index()
plt.figure(figsize=(10, 6))
plt.bar(hour_season_counts.index, hour_season_counts.values, color='skyblue')
plt.title('Total Daily Tenant Based on Season')
plt.xlabel('Season')
plt.ylabel('Total Tenant')
plt.xticks(hour_season_counts.index)
st.pyplot(plt)

# Monthly Counts
hour_monthly_cnt = df_hour.groupby(by="mnth")[['casual', 'registered', 'cnt']].sum().sort_index()
plt.figure(figsize=(10, 6))
months = hour_monthly_cnt.index
bar_width = 0.35
index = np.arange(len(months))
plt.bar(index - bar_width/2, hour_monthly_cnt['cnt'], bar_width, label='Total', color='skyblue')
plt.title('Total Hourly Tenant Based on Month')
plt.xlabel('Month')
plt.ylabel('Tenant (cnt)')
plt.xticks(index, months)
plt.legend()
st.pyplot(plt)

# EDA: Daily Data
st.subheader("Explore Daily Data")

# Change season values for daily data
df_day['season'] = df_day['season'].replace(season_mapping)

# Total daily tenant based on season
day_season_counts = df_day.groupby(by="season")['cnt'].sum().sort_index()
plt.figure(figsize=(10, 6))
plt.bar(day_season_counts.index, day_season_counts.values, color='skyblue')
plt.title('Total Daily Tenant Based on Season')
plt.xlabel('Season')
plt.ylabel('Total Tenant')
plt.xticks(day_season_counts.index)
st.pyplot(plt)

# Monthly Counts for daily data
day_monthly_cnt = df_day.groupby(by="mnth")[['casual', 'registered', 'cnt']].sum().sort_index()
plt.figure(figsize=(10, 6))
months = day_monthly_cnt.index
bar_width = 0.35
index = np.arange(len(months))
plt.bar(index - bar_width/2, day_monthly_cnt['cnt'], bar_width, label='Total', color='skyblue')
plt.title('Total Daily Tenant Based on Month')
plt.xlabel('Month')
plt.ylabel('Tenant (cnt)')
plt.xticks(index, months)
plt.legend()
st.pyplot(plt)

# Merging datasets
df_hour_agg = df_hour.groupby('dteday').agg({
    'casual': 'sum',
    'registered': 'sum',
    'cnt': 'sum'
}).reset_index()

df_combined = pd.merge(df_day, df_hour_agg, on='dteday', suffixes=('_day', '_hour'))

# Seasonal and monthly trend
st.subheader("Seasonal and monthly trend")
total_by_season_mnth = df_combined.groupby(['season', 'mnth']).agg({
    'cnt_day': 'sum',
    'cnt_hour': 'sum'
}).reset_index()

plt.figure(figsize=(12, 8))
sns.barplot(data=total_by_season_mnth, x='mnth', y='cnt_day', hue='season')
plt.title('Total Penyewaan Sepeda Berdasarkan Musim (Season) dan Bulan')
plt.xlabel('Month')
plt.ylabel('Total Penyewaan')
plt.xticks(rotation=45)
st.pyplot(plt)

# New Graph: Total Sharings Based on Season and Year
st.subheader("Total Sharings Based on Season and Year")
total_by_season_year = df_combined.groupby(['yr', 'season']).agg({
    'cnt_day': 'sum',
    'cnt_hour': 'sum'
}).reset_index()
total_by_season_year['yr'] = total_by_season_year['yr'].replace({0: 2011, 1: 2012})

plt.figure(figsize=(10, 6))
sns.barplot(data=total_by_season_year, x='season', y='cnt_day', hue='yr')
plt.title('Total Penyewaan Sepeda Berdasarkan Musim Tahun 2011 dan 2012')
plt.xlabel('Season')
plt.ylabel('Total Penyewaan (Hari)')
plt.legend(title='Year', loc='upper right', labels=['2011', '2012'])
st.pyplot(plt)

# Analysis on Holidays
st.subheader("Holiday Analysis")
df_combined['day_info'] = df_combined.apply(
    lambda row: f"{'Holiday' if row['holiday'] == 1 else 'Non-Holiday'}",
    axis=1
)

holiday_grouped = df_combined.groupby(['holiday']).agg({
    'cnt_day': 'sum',
    'cnt_hour': 'sum',
}).reset_index()

holiday_grouped['holiday_label'] = holiday_grouped['holiday'].replace({0: 'Non-Holiday', 1: 'Holiday'})

# Visualize holiday analysis
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='holiday_label', y='cnt_day', data=holiday_grouped, palette='viridis', legend=False)
plt.title('Count of Days')
plt.xlabel('Holiday Type')
plt.ylabel('Count of Days')

plt.subplot(1, 2, 2)
sns.barplot(x='holiday_label', y='cnt_hour', data=holiday_grouped, palette='viridis', legend=False)
plt.title('Count of Hours')
plt.xlabel('Holiday Type')
plt.ylabel('Count of Hours')

plt.tight_layout()
st.pyplot(plt)

# Clustering Analysis
st.subheader("Clustering Analysis")
features = df_hour[['casual', 'registered']].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

centroid_1 = features_scaled[0]
centroid_2 = features_scaled[1]

def euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

clusters = []
for point in features_scaled:
    dist_to_centroid_1 = euclidean_distance(point, centroid_1)
    dist_to_centroid_2 = euclidean_distance(point, centroid_2)

    if dist_to_centroid_1 < dist_to_centroid_2:
        clusters.append(0)  # Cluster 1
    else:
        clusters.append(1)  # Cluster 2

df_hour['cluster'] = clusters

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_hour, x='casual', y='registered', hue='cluster', palette='viridis')
plt.title('Clustering Hasil Penyewaan Sepeda')
plt.xlabel('Jumlah Casual')
plt.ylabel('Jumlah Registered')
plt.legend(title='Cluster')
st.pyplot(plt)

# Save combined DataFrame to CSV
df_combined.to_csv("main_data.csv", index=False)

# Conclusion Section
st.subheader("Conclusions")
st.write("""
### Question 1 Conclusion
- Significant differences exist in bike rental trends across seasons.
- Rental behavior is influenced by seasonal factors.
- Insights can guide marketing strategies for bike rental services.

### Question 2 Conclusion
- Holiday vs. Non-Holiday rental trends show significant differences.
- Higher Sharings occur on weekdays, especially Thursday to Saturday.
- Service providers can optimize operations based on demand trends.
""")
