import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
import shap
# -----------------------------
# Load Data & Models
# -----------------------------

df = pd.read_csv("employee_salary_dataset.csv")

import joblib
model = joblib.load("models/salary_model.pkl")
kmeans_pipeline = joblib.load("models/kmeans_model.pkl")



# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(page_title="Workforce Intelligence Platform", layout="wide")

st.title("💼 Workforce Intelligence & Salary Prediction Platform")
st.markdown("Advanced ML-based Salary Modeling & Employee Segmentation")

# -----------------------------
# Sidebar Input
# -----------------------------

st.sidebar.header("Enter Employee Details")

age = st.sidebar.number_input("Age", 18, 65, 30)
experience = st.sidebar.number_input("Experience (Years)", 0, 40, 5)
hours = st.sidebar.number_input("Hours per Week", 10, 80, 40)

education = st.sidebar.selectbox("Education Level", df["Education Level"].unique())
industry = st.sidebar.selectbox("Industry", df["Industry"].unique())
work_mode = st.sidebar.selectbox("Work Mode", df["Work Mode"].unique())

skills_input = st.sidebar.text_input("Skills (comma separated)", "Python, ML")

# -----------------------------
# Prediction
# -----------------------------

if st.sidebar.button("Predict Salary"):

    input_df = pd.DataFrame({
        "Age": [age],
        "Education Level": [education],
        "Job Title": ["Unknown"],
        "Experience (Years)": [experience],
        "Industry": [industry],
        "Hours/Week": [hours],
        "Work Mode": [work_mode],
        "Skills": [skills_input]
    })

    # -------------------
    # Prediction
    # -------------------

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Salary: ${int(prediction):,}")

    # -------------------
    # SHAP Explanation
    # -------------------

    
# Feature Importance Section
# -----------------------------

st.subheader("📊 Feature Importance")

rf_model = model.named_steps["model"]

if hasattr(rf_model, "feature_importances_"):
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.invert_yaxis()
    st.pyplot(fig)

# -----------------------------
# Skill Salary Insights
# -----------------------------

st.subheader("🔥 Top Paying Skills")

df["Skills_List"] = df["Skills"].str.split(", ")
skills_exploded = df.explode("Skills_List")

skill_salary = skills_exploded.groupby("Skills_List")["Salary (USD)"].mean().sort_values(ascending=False).head(10)

st.bar_chart(skill_salary)

# -----------------------------
# Clustering Visualization
# -----------------------------

st.subheader("🧠 Employee Clusters & Segmentation")

X = df.drop("Salary (USD)", axis=1)

# Predict clusters using pipeline
clusters = kmeans_pipeline.predict(X)

# Get processed numeric data
X_processed = kmeans_pipeline.named_steps["preprocess"].transform(X)

# PCA for 2D visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(X_processed)

cluster_df = pd.DataFrame({
    "PC1": reduced[:, 0],
    "PC2": reduced[:, 1],
    "Cluster": clusters
})

# Add salary to cluster dataframe
cluster_df["Salary"] = df["Salary (USD)"]

# --------------------------------
# Auto Cluster Interpretation
# --------------------------------

cluster_salary = cluster_df.groupby("Cluster")["Salary"].mean().sort_values()

cluster_mapping = {}
labels = ["Early Career", "Mid Level", "Senior Professionals"]

for cluster_id, label in zip(cluster_salary.index, labels):
    cluster_mapping[cluster_id] = label

cluster_df["Segment"] = cluster_df["Cluster"].map(cluster_mapping)

# --------------------------------
# Interactive Plotly Chart
# --------------------------------

fig = px.scatter(
    cluster_df,
    x="PC1",
    y="PC2",
    color="Segment",
    hover_data=["Salary"],
    title="Employee Segmentation Map"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------
# Cluster Salary Insights
# --------------------------------

st.subheader("📊 Cluster Salary Insights")

cluster_summary = cluster_df.groupby("Segment")["Salary"].agg(["mean", "count"]).reset_index()
cluster_summary.columns = ["Segment", "Average Salary", "Employee Count"]

st.dataframe(cluster_summary)

st.bar_chart(cluster_summary.set_index("Segment")["Average Salary"])