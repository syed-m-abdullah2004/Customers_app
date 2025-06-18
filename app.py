import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set page config
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# Title and description
st.title("Customer Segmentation Dashboard")
st.markdown("""
This app performs customer segmentation using K-Means clustering on synthetic customer data.
Adjust the parameters in the sidebar to explore different clustering configurations.
""")

# Sidebar controls
with st.sidebar:
    st.header("Clustering Parameters")
    n_clusters = st.slider("Number of clusters", 2, 10, 3)
    scaling_method = st.selectbox("Scaling method", ["StandardScaler", "MinMaxScaler"])
    show_raw_data = st.checkbox("Show raw data", value=True)
    show_elbow_plot = st.checkbox("Show elbow method plot", value=True)
    show_cluster_stats = st.checkbox("Show cluster statistics", value=True)

# Generate synthetic data
@st.cache_data
def generate_data():
    np.random.seed(42)
    num_customers = 2000
    customer_data = {
        "CustomerID": np.arange(1, num_customers + 1),
        "Age": np.random.randint(18, 65, size=num_customers),
        "AverageSpend": np.round(np.random.uniform(10, 500, size=num_customers), 2),
        "VisitsPerWeek": np.random.randint(1, 8, size=num_customers),
        "PromotionInterest": np.random.randint(1, 11, size=num_customers)
    }
    return pd.DataFrame(customer_data)

df = generate_data()
X = df.drop('CustomerID', axis=1)

# Show raw data if selected
if show_raw_data:
    st.subheader("Raw Customer Data")
    st.dataframe(df.head())

# Main clustering analysis
st.subheader("Clustering Analysis")

# Create pipeline based on user selection
if scaling_method == "StandardScaler":
    scaler = StandardScaler()
else:
    scaler = MinMaxScaler()

pipe = Pipeline([
    ('scaler', scaler),
    ('kmeans', KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
])

# Perform clustering
df['Cluster'] = pipe.fit_predict(X)

# Assign cluster names
cluster_names = {i: f'Group {i+1}' for i in range(n_clusters)}
df['Segment'] = df['Cluster'].map(cluster_names)

# Calculate silhouette score
silhouette_avg = silhouette_score(X, df['Cluster'])

# Display metrics
col1, col2 = st.columns(2)
with col1:
    st.metric("Number of customers", len(df))
with col2:
    st.metric("Silhouette Score", f"{silhouette_avg:.4f}")

# Show cluster distribution
st.subheader("Cluster Distribution")
cluster_counts = df['Segment'].value_counts()
st.bar_chart(cluster_counts)

# Show elbow method plot if selected
if show_elbow_plot:
    st.subheader("Elbow Method Analysis")
    ssc = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        if k > 1:
            ssc.append(kmeans.inertia_)
    
    fig, ax = plt.subplots()
    ax.plot(range(2, 11), ssc, marker='o')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')
    st.pyplot(fig)

# Show cluster statistics if selected
if show_cluster_stats:
    st.subheader("Cluster Characteristics")
    cluster_stats = df.groupby('Segment').agg({
        'Age': 'mean',
        'AverageSpend': 'mean',
        'VisitsPerWeek': 'mean',
        'PromotionInterest': 'mean'
    }).round(2)
    st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'))

# Cluster visualization
st.subheader("Cluster Visualization")

# Feature selection for plotting
feature_pairs = [
    ('Age', 'AverageSpend'),
    ('VisitsPerWeek', 'AverageSpend'),
    ('Age', 'PromotionInterest'),
    ('VisitsPerWeek', 'PromotionInterest')
]

cols = st.columns(2)
for i, (x_feat, y_feat) in enumerate(feature_pairs):
    with cols[i % 2]:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            df[x_feat], 
            df[y_feat], 
            c=df['Cluster'], 
            cmap='viridis',
            alpha=0.6
        )
        ax.set_xlabel(x_feat)
        ax.set_ylabel(y_feat)
        ax.set_title(f"{x_feat} vs {y_feat}")
        plt.colorbar(scatter, label='Cluster')
        st.pyplot(fig)

# Download results
st.subheader("Download Results")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download segmented data as CSV",
    data=csv,
    file_name='customer_segments.csv',
    mime='text/csv'
)

# Footer
st.markdown("---")
st.markdown("Customer Segmentation Dashboard - © 2023")
