# streamlit_app.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score
)

import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS (Makes Everything Bold & Clear)
# -------------------------------------------------
st.markdown("""
<style>
.big-font {
    font-size:26px !important;
    font-weight:bold !important;
}
.metric-value {
    font-size:32px !important;
    font-weight:900 !important;
    color:#1f77b4 !important;
}
.section-header {
    background-color:#e8f5e9;
    padding:12px;
    border-radius:10px;
    font-size:28px;
    font-weight:800;
    color:#2e7d32;
}
.sidebar-header {
    font-size:20px;
    font-weight:800;
    color:#1565c0;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Main Title
# -------------------------------------------------
st.markdown(
    "<h1 style='text-align:center; color:#2E8B57; font-size:42px;'>🌱 Dry Bean Classification App</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.markdown("<p class='sidebar-header'>📂 Upload & Model Selection</p>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "📁 Upload Test CSV File",
    type=["csv"]
)

model_options = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]

selected_model = st.sidebar.selectbox(
    "🤖 Select Model",
    model_options
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Download Sample Test CSV**")
github_test_link = "https://github.com/gchandramohan/ML_Assignment/blob/main/test_data/dry_bean_test.csv"
st.sidebar.markdown(f"[Click here to download]({github_test_link})")

# -------------------------------------------------
# Load Models
# -------------------------------------------------
models = {
    "Logistic Regression": pickle.load(open("model/logistic.pkl", "rb")),
    "Decision Tree": pickle.load(open("model/decision_tree.pkl", "rb")),
    "KNN": pickle.load(open("model/knn.pkl", "rb")),
    "Naive Bayes": pickle.load(open("model/naive_bayes.pkl", "rb")),
    "Random Forest": pickle.load(open("model/random_forest.pkl", "rb")),
    "XGBoost": pickle.load(open("model/xgboost.pkl", "rb")),
}

scaler = pickle.load(open("model/scaler.pkl", "rb"))
label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))

# -------------------------------------------------
# Main Content
# -------------------------------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "Class" not in df.columns:
        st.error("❌ CSV must contain 'Class' column.")
        st.stop()

    df = df.dropna(subset=["Class"])
    df["Class"] = df["Class"].astype(str).str.strip()

    y_true = df["Class"]

    unseen_labels = set(y_true.unique()) - set(label_encoder.classes_)
    if len(unseen_labels) > 0:
        st.error(f"❌ Unseen labels found: {unseen_labels}")
        st.stop()

    y_true = label_encoder.transform(y_true)

    X = df.drop("Class", axis=1)
    X_scaled = scaler.transform(X)

    model = models[selected_model]
    y_pred = model.predict(X_scaled)

    # -------------------------------------------------
    # Evaluation Metrics Section
    # -------------------------------------------------
    st.markdown("<div class='section-header'>Evaluation Metrics</div>", unsafe_allow_html=True)
    st.markdown(" ")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_true, y_pred)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
    else:
        auc = None

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<p class='big-font'>Accuracy</p>", unsafe_allow_html=True)
    col1.markdown(f"<p class='metric-value'>{acc:.4f}</p>", unsafe_allow_html=True)

    col2.markdown(f"<p class='big-font'>Precision</p>", unsafe_allow_html=True)
    col2.markdown(f"<p class='metric-value'>{prec:.4f}</p>", unsafe_allow_html=True)

    col3.markdown(f"<p class='big-font'>Recall</p>", unsafe_allow_html=True)
    col3.markdown(f"<p class='metric-value'>{rec:.4f}</p>", unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    col4.markdown(f"<p class='big-font'>F1 Score</p>", unsafe_allow_html=True)
    col4.markdown(f"<p class='metric-value'>{f1:.4f}</p>", unsafe_allow_html=True)

    col5.markdown(f"<p class='big-font'>MCC</p>", unsafe_allow_html=True)
    col5.markdown(f"<p class='metric-value'>{mcc:.4f}</p>", unsafe_allow_html=True)

    col6.markdown(f"<p class='big-font'>AUC</p>", unsafe_allow_html=True)
    col6.markdown(f"<p class='metric-value'>{auc:.4f}" if auc else "N/A", unsafe_allow_html=True)

    # -------------------------------------------------
    # Confusion Matrix Section
    # -------------------------------------------------
    st.markdown(" ")
    st.markdown("<div class='section-header'>Confusion Matrix</div>", unsafe_allow_html=True)
    st.markdown(" ")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        linewidths=0.5,
        ax=ax
    )

    ax.set_xlabel("Predicted Label", fontsize=14, fontweight='bold')
    ax.set_ylabel("True Label", fontsize=14, fontweight='bold')
    ax.set_title(f"Confusion Matrix - {selected_model}", fontsize=16, fontweight='bold')

    st.pyplot(fig)

else:
    st.info("Please upload a test CSV file from the sidebar to begin.")
