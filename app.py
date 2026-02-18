import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Customer Analytics", layout="wide")
st.title("🛍️ Interactive Customer Behavior Analytics")

# ==================================================
# SAMPLE DATA
# ==================================================
np.random.seed(42)  # For reproducibility

mall_df = pd.DataFrame({
    "CustomerID": range(1, 201),
    "Gender": np.random.choice(["Male", "Female"], 200),
    "Age": np.random.randint(18, 70, 200),
    "Annual Income (k$)": np.random.randint(15, 150, 200),
    "Spending Score (1-100)": np.random.randint(1, 100, 200)
})

ratings_df = pd.DataFrame({
    "userId": np.random.randint(1, 21, 200),
    "movieId": np.random.randint(1, 31, 200),
    "rating": np.random.randint(1, 6, 200),
    "timestamp": np.random.randint(1000000000, 2000000000, 200)
})

features = mall_df[["Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# ==================================================
# SIDEBAR MENU
# ==================================================
option = st.sidebar.selectbox(
    "Select Task",
    [
        "Data Overview",
        "Customer Segmentation (K-Means Animated)",
        "Anomaly Detection",
        "PCA Visualization",
        "Recommendation System"
    ]
)

# ==================================================
# DATA OVERVIEW
# ==================================================
if option == "Data Overview":
    st.subheader("Mall Dataset Preview")
    st.dataframe(mall_df)
    col1, col2 = st.columns(2)
    col1.metric("Number of Customers", mall_df.shape[0])
    col2.metric("Number of Features", mall_df.shape[1])
    st.subheader("Gender Distribution")
    fig = px.pie(mall_df, names="Gender", color="Gender")
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# K-MEANS ANIMATION
# ==================================================
if option == "Customer Segmentation (K-Means Animated)":
    st.subheader("Animated K-Means Clustering")

    k = st.slider("Select Number of Clusters", 2, 8, 4)

    # Initialize centroids randomly from data points
    rng = np.random.default_rng(42)
    centroids = scaled_features[rng.choice(len(scaled_features), k, replace=False)]

    frames = []
    df_plot = pd.DataFrame(scaled_features, columns=["X", "Y"])
    df_plot["CustomerID"] = mall_df["CustomerID"]

    for iteration in range(10):  # 10 iterations of K-Means
        # Compute distances and assign clusters
        distances = np.linalg.norm(df_plot[["X", "Y"]].values[:, None] - centroids[None, :], axis=2)
        df_plot["Cluster"] = np.argmin(distances, axis=1)

        # Save frame for animation
        df_frame = df_plot.copy()
        df_frame["Iteration"] = iteration
        frames.append(df_frame)

        # Update centroids
        for i in range(k):
            points_in_cluster = df_plot[df_plot["Cluster"] == i][["X", "Y"]]
            if len(points_in_cluster) > 0:
                centroids[i] = points_in_cluster.mean().values

    full_df = pd.concat(frames)

    fig = px.scatter(
        full_df,
        x="X",
        y="Y",
        animation_frame="Iteration",
        color="Cluster",
        hover_data=["CustomerID"],
        title="Live K-Means Clustering Animation",
        color_continuous_scale=px.colors.qualitative.Bold,
        range_x=[-3, 3],
        range_y=[-3, 3]
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# ANOMALY DETECTION
# ==================================================
if option == "Anomaly Detection":
    st.subheader("Isolation Forest Anomaly Detection")

    iso = IsolationForest(contamination=0.05, random_state=42)
    mall_df["Anomaly"] = iso.fit_predict(scaled_features)
    mall_df["Anomaly_Label"] = np.where(mall_df["Anomaly"] == -1, "Anomaly", "Normal")

    fig = px.scatter(
        mall_df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Anomaly_Label",
        hover_data=["CustomerID", "Age", "Gender"],
        color_discrete_map={"Normal": "blue", "Anomaly": "red"},
        title="Anomaly Detection"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# PCA VISUALIZATION
# ==================================================
if option == "PCA Visualization":
    st.subheader("PCA Dimensionality Reduction")

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_features)
    mall_df["PC1"] = components[:, 0]
    mall_df["PC2"] = components[:, 1]

    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

    fig = px.scatter(
        mall_df,
        x="PC1",
        y="PC2",
        color="Cluster" if "Cluster" in mall_df.columns else None,
        hover_data=["CustomerID", "Age", "Gender"],
        title="PCA Projection"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# RECOMMENDATION SYSTEM
# ==================================================
if option == "Recommendation System":
    st.subheader("User-Based Collaborative Filtering")

    user_item = ratings_df.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    similarity = cosine_similarity(user_item)
    similarity_df = pd.DataFrame(similarity, index=user_item.index, columns=user_item.index)

    def recommend(user_id, n=5):
        if user_id not in similarity_df.index:
            return "User not found."
        similar_users = similarity_df[user_id].sort_values(ascending=False)[1:6]
        weighted_ratings = user_item.loc[similar_users.index].mean()
        already_rated = user_item.loc[user_id]
        recommendations = weighted_ratings[already_rated == 0]
        return recommendations.sort_values(ascending=False).head(n)

    max_user = int(ratings_df["userId"].max())
    user_input = st.number_input("Enter User ID", min_value=1, max_value=max_user, value=1)

    if st.button("Get Recommendations"):
        st.write(recommend(user_input))
