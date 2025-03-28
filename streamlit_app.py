import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, 
    precision_recall_curve, auc, f1_score, accuracy_score, 
    precision_score, recall_score
)
from sklearn.decomposition import PCA

# Try to import SMOTE, but provide a fallback if it's not available
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    st.warning("The imblearn package is not installed. SMOTE functionality will be disabled.")

import io
import base64
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection", 
    page_icon="💳", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #FF5757;}
    .sub-header {font-size: 1.5rem; margin-bottom: 1rem;}
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #FF5757;
    }
    .fraud-alert {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .genuine-alert {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton button {background-color: #FF5757; color: white;}
    .feature-importance {margin-top: 2rem;}
    .threshold-slider {margin-top: 2rem; margin-bottom: 2rem;}
    .chart-container {margin-top: 2rem;}
    .footer {text-align: center; margin-top: 3rem; color: #888;}
    .loading-spinner {text-align: center; margin-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.5
if 'df' not in st.session_state:
    st.session_state.df = None
if 'fraud_indices' not in st.session_state:
    st.session_state.fraud_indices = None
if 'genuine_indices' not in st.session_state:
    st.session_state.genuine_indices = None

# Sidebar
with st.sidebar:
    st.title("💳 Fraud Detection")
    st.markdown("---")
    
    st.markdown("### Data Options")
    data_option = st.radio(
        "Choose data source:",
        ["Upload CSV file", "Use sample dataset"]
    )
    
    if data_option == "Upload CSV file":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    st.markdown("### Model Parameters")
    n_estimators = st.slider("Number of trees", 50, 300, 100, 10)
    max_depth = st.slider("Maximum tree depth", 5, 30, 10, 1)
    min_samples_split = st.slider("Minimum samples to split", 2, 20, 2, 1)
    
    st.markdown("### Preprocessing Options")
    scaling_method = st.selectbox(
        "Scaling method",
        ["Standard Scaler", "Robust Scaler", "No Scaling"]
    )
    
    # Only show SMOTE option if it's available
    if SMOTE_AVAILABLE:
        handle_imbalance = st.checkbox("Handle class imbalance with SMOTE", True)
    else:
        handle_imbalance = False
        st.info("SMOTE is not available. Install 'imbalanced-learn' package to enable this feature.")
    
    use_pca = st.checkbox("Apply PCA for dimensionality reduction", False)
    if use_pca:
        n_components = st.slider("Number of PCA components", 2, 20, 10, 1)
    
    st.markdown("### Visualization Options")
    plot_style = st.selectbox(
        "Plot style",
        ["Plotly", "Matplotlib"]
    )
    
    st.markdown("---")
    st.markdown("Made by Sai Rupa Jhade")

# Main content
st.markdown('<p class="main-header">💳 Credit Card Fraud Detection</p>', unsafe_allow_html=True)
st.markdown("""
This interactive dashboard analyzes credit card transactions to detect potential frauds using machine learning.
Explore the dataset, train models with custom parameters, and test real-time predictions.
""")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Data Exploration", "Model Training", "Real-time Detection", "Performance Analysis"])

# Function to load data
@st.cache_data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
    else:
        # Generate synthetic data if no file is provided
        np.random.seed(42)
        n_samples = 10000
        n_features = 30
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target (imbalanced - 1% fraud)
        y = np.zeros(n_samples)
        fraud_indices = np.random.choice(range(n_samples), size=int(n_samples * 0.01), replace=False)
        y[fraud_indices] = 1
        
        # Create time and amount features
        time = np.random.randint(0, 172800, n_samples)
        amount = np.random.exponential(scale=100, size=n_samples)
        # Make fraudulent transactions have slightly higher amounts on average
        amount[fraud_indices] = amount[fraud_indices] * 1.5
        
        # Combine into dataframe - ensure feature_names matches the number of columns in X
        feature_names = [f"V{i}" for i in range(1, n_features+1)]
        
        # Create DataFrame with the correct number of columns
        df = pd.DataFrame(X, columns=feature_names[:n_features])
        
        # Add time, amount and class columns
        df['Time'] = time
        df['Amount'] = amount
        df['Class'] = y.astype(int)
    
    return df

# Function to preprocess data
def preprocess_data(df, scaling_method, handle_imbalance, use_pca=False, n_components=10):
    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    # Save feature names
    feature_names = X.columns.tolist()
    
    # Scale features
    if scaling_method == "Standard Scaler":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaling_method == "Robust Scaler":
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = None
        X_scaled = X.values
    
    # Apply PCA if selected
    if use_pca:
        pca = PCA(n_components=n_components)
        X_scaled = pca.fit_transform(X_scaled)
        feature_names = [f"PC{i+1}" for i in range(n_components)]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    
    # Handle class imbalance if selected and SMOTE is available
    if handle_imbalance and SMOTE_AVAILABLE:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    return X_train, X_test, y_train, y_test, scaler, feature_names

# Function to train model
def train_model(X_train, y_train, n_estimators, max_depth, min_samples_split):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    clf.fit(X_train, y_train)
    
    return clf

# Function to evaluate model
def evaluate_model(model, X_test, y_test, threshold=0.5):
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "confusion_matrix": cm,
        "y_proba": y_proba,
        "y_pred": y_pred
    }

# Function to plot confusion matrix
def plot_confusion_matrix(cm, plot_style="Plotly"):
    if plot_style == "Plotly":
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Genuine", "Fraud"],
            y=["Genuine", "Fraud"],
            colorscale="Reds",
            showscale=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=400,
            height=400
        )
        
        # Add annotations
        annotations = []
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=str(cm[i, j]),
                        showarrow=False,
                        font=dict(color="white" if cm[i, j] > cm.max() / 2 else "black")
                    )
                )
        
        fig.update_layout(annotations=annotations)
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        ax.set_xticklabels(["Genuine", "Fraud"])
        ax.set_yticklabels(["Genuine", "Fraud"])
        return fig

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, plot_style="Plotly"):
    if plot_style == "Plotly":
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"ROC Curve (AUC = {roc_auc:.3f})",
            line=dict(color="red", width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color="gray", width=2, dash="dash")
        ))
        
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=500,
            height=400,
            showlegend=True
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color="red", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        return fig

# Function to plot Precision-Recall curve
def plot_pr_curve(recall_curve, precision_curve, pr_auc, plot_style="Plotly"):
    if plot_style == "Plotly":
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall_curve, y=precision_curve,
            mode="lines",
            name=f"PR Curve (AUC = {pr_auc:.3f})",
            line=dict(color="blue", width=2)
        ))
        
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            width=500,
            height=400,
            showlegend=True
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(recall_curve, precision_curve, color="blue", lw=2, label=f"PR Curve (AUC = {pr_auc:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        return fig

# Function to plot feature importance
def plot_feature_importance(model, feature_names, plot_style="Plotly", top_n=10):
    # Get feature importance
    importances = model.feature_importances_
    
    # Sort feature importance
    indices = np.argsort(importances)[::-1]
    
    # Get top N features
    top_indices = indices[:min(top_n, len(feature_names))]
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    
    if plot_style == "Plotly":
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_importances,
            y=top_features,
            orientation="h",
            marker=dict(color="rgba(255, 87, 87, 0.8)")
        ))
        
        fig.update_layout(
            title=f"Top {len(top_features)} Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400,
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top_features, top_importances, color="red", alpha=0.8)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top {len(top_features)} Feature Importance")
        ax.invert_yaxis()
        return fig

# Function to plot threshold analysis
def plot_threshold_analysis(y_test, y_proba, plot_style="Plotly"):
    thresholds = np.arange(0.1, 1.0, 0.05)
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
    
    if plot_style == "Plotly":
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=precision_scores,
            mode="lines+markers",
            name="Precision",
            line=dict(color="blue", width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=recall_scores,
            mode="lines+markers",
            name="Recall",
            line=dict(color="red", width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=f1_scores,
            mode="lines+markers",
            name="F1 Score",
            line=dict(color="green", width=2)
        ))
        
        fig.update_layout(
            title="Threshold Analysis",
            xaxis_title="Threshold",
            yaxis_title="Score",
            width=700,
            height=400,
            showlegend=True
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thresholds, precision_scores, "b-", label="Precision", marker="o")
        ax.plot(thresholds, recall_scores, "r-", label="Recall", marker="o")
        ax.plot(thresholds, f1_scores, "g-", label="F1 Score", marker="o")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Threshold Analysis")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        return fig

# Function to generate a random transaction
def generate_random_transaction(fraud_probability=0.5):
    # Generate random features
    features = np.random.randn(30)
    
    # Determine if transaction is fraudulent based on probability
    is_fraud = np.random.random() < fraud_probability
    
    # Modify features to make them more like fraud if needed
    if is_fraud:
        # Add some bias to make it look more like fraud
        features[0] += 2.0  # Assuming V1 is important for fraud detection
        features[2] -= 1.5  # Assuming V3 is important for fraud detection
        amount = np.random.exponential(scale=150)  # Higher amount for fraud
    else:
        amount = np.random.exponential(scale=100)  # Normal amount
    
    # Add time and amount
    time = np.random.randint(0, 172800)
    
    # Create transaction dictionary
    transaction = {}
    for i in range(28):
        transaction[f"V{i+1}"] = features[i]
    
    transaction["Time"] = time
    transaction["Amount"] = amount
    
    return transaction, is_fraud

# Tab 1: Data Exploration
with tab1:
    st.markdown('<p class="sub-header">📊 Data Exploration</p>', unsafe_allow_html=True)
    
    # Load data
    if data_option == "Upload CSV file" and uploaded_file is not None:
        df = load_data(uploaded_file)
        st.session_state.df = df
    elif data_option == "Use sample dataset" or "df" not in st.session_state or st.session_state.df is None:
        df = load_data()
        st.session_state.df = df
    else:
        df = st.session_state.df
    
    # Display dataset info
    st.markdown("### Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    with col2:
        fraud_count = df["Class"].sum()
        st.metric("Fraudulent Transactions", f"{fraud_count:,} ({fraud_count/len(df)*100:.2f}%)")
    with col3:
        st.metric("Genuine Transactions", f"{len(df) - fraud_count:,} ({(len(df) - fraud_count)/len(df)*100:.2f}%)")
    
    # Data preview with pagination
    st.markdown("### Data Preview")
    page_size = st.slider("Rows per page", 5, 50, 10)
    page_number = st.number_input("Page", min_value=1, max_value=max(1, len(df) // page_size), step=1)
    
    start_idx = (page_number - 1) * page_size
    end_idx = min(start_idx + page_size, len(df))
    
    st.dataframe(df.iloc[start_idx:end_idx])
    
    # Class distribution
    st.markdown("### Class Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if plot_style == "Plotly":
            fig = px.pie(
                names=["Genuine", "Fraud"],
                values=[len(df) - fraud_count, fraud_count],
                title="Transaction Distribution",
                color_discrete_sequence=["#4CAF50", "#F44336"]
            )
            st.plotly_chart(fig)
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(
                [len(df) - fraud_count, fraud_count],
                labels=["Genuine", "Fraud"],
                autopct='%1.1f%%',
                colors=["#4CAF50", "#F44336"],
                startangle=90
            )
            ax.axis('equal')
            ax.set_title("Transaction Distribution")
            st.pyplot(fig)
    
    with col2:
        # Store fraud and genuine indices for later use
        fraud_indices = df[df["Class"] == 1].index.tolist()
        genuine_indices = df[df["Class"] == 0].index.tolist()
        
        st.session_state.fraud_indices = fraud_indices
        st.session_state.genuine_indices = genuine_indices
        
        st.markdown("""
        ### Class Imbalance
        
        Credit card fraud detection typically deals with highly imbalanced datasets where fraudulent transactions are rare.
        
        **Challenges:**
        - Models tend to be biased towards the majority class
        - Standard accuracy is not a good metric
        - Need to focus on precision and recall
        
        **Solutions:**
        - Resampling techniques (SMOTE)
        - Adjusting class weights
        - Using appropriate evaluation metrics
        """)
    
    # Feature analysis
    st.markdown("### Feature Analysis")
    
    # Amount distribution
    st.subheader("Transaction Amount Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if plot_style == "Plotly":
            fig = px.histogram(
                df, x="Amount", color="Class",
                color_discrete_map={0: "#4CAF50", 1: "#F44336"},
                labels={"Class": "Transaction Type", "Amount": "Amount"},
                title="Amount Distribution by Class",
                marginal="box"
            )
            fig.update_layout(xaxis_title="Amount", yaxis_title="Count")
            st.plotly_chart(fig)
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(data=df, x="Amount", hue="Class", bins=50, ax=ax)
            ax.set_xlabel("Amount")
            ax.set_ylabel("Count")
            ax.set_title("Amount Distribution by Class")
            ax.legend(["Genuine", "Fraud"])
            st.pyplot(fig)
    
    with col2:
        # Amount statistics
        st.markdown("### Amount Statistics")
        
        amount_stats = df.groupby("Class")["Amount"].agg(["mean", "median", "min", "max"]).reset_index()
        amount_stats.columns = ["Class", "Mean", "Median", "Min", "Max"]
        amount_stats["Class"] = amount_stats["Class"].map({0: "Genuine", 1: "Fraud"})
        
        st.dataframe(amount_stats)
        
        st.markdown("""
        **Observations:**
        - Fraudulent transactions often have different amount patterns
        - Some fraudsters test small amounts before making larger transactions
        - Amount alone is not sufficient for fraud detection
        """)
    
    # Time analysis
    st.subheader("Transaction Time Analysis")
    
    # Convert time to hours
    df_time = df.copy()
    df_time["Hour"] = df_time["Time"] / 3600  # Convert seconds to hours
    
    if plot_style == "Plotly":
        fig = px.histogram(
            df_time, x="Hour", color="Class",
            color_discrete_map={0: "#4CAF50", 1: "#F44336"},
            labels={"Class": "Transaction Type", "Hour": "Hour of Day"},
            title="Transaction Distribution by Hour",
            nbins=24
        )
        fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Count")
        st.plotly_chart(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=df_time, x="Hour", hue="Class", bins=24, ax=ax)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Count")
        ax.set_title("Transaction Distribution by Hour")
        ax.legend(["Genuine", "Fraud"])
        st.pyplot(fig)
    
    # Feature correlation
    st.subheader("Feature Correlation")
    
    # Select features for correlation analysis
    correlation_features = st.multiselect(
        "Select features for correlation analysis",
        df.columns.tolist(),
        default=["V1", "V2", "V3", "V4", "Amount", "Class"]
    )
    
    if correlation_features:
        if plot_style == "Plotly":
            corr_matrix = df[correlation_features].corr()
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale="RdBu_r",
                title="Correlation Matrix"
            )
            st.plotly_chart(fig)
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df[correlation_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", ax=ax)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)
    
    # PCA visualization if selected
    if use_pca:
        st.subheader("PCA Visualization")
        
        # Perform PCA
        X = df.drop("Class", axis=1)
        y = df["Class"]
        
        if scaling_method == "Standard Scaler":
            scaler = StandardScaler()
        elif scaling_method == "Robust Scaler":
            scaler = RobustScaler()
        else:
            scaler = None
        
        if scaler:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["Class"] = y.values
        
        if plot_style == "Plotly":
            fig = px.scatter(
                pca_df, x="PC1", y="PC2", color="Class",
                color_discrete_map={0: "#4CAF50", 1: "#F44336"},
                labels={"Class": "Transaction Type"},
                title="PCA Visualization"
            )
            st.plotly_chart(fig)
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Class", ax=ax)
            ax.set_title("PCA Visualization")
            ax.legend(["Genuine", "Fraud"])
            st.pyplot(fig)
        
        # Explained variance
        explained_variance = pca.explained_variance_ratio_
        st.write(f"Explained variance: PC1 = {explained_variance[0]:.2%}, PC2 = {explained_variance[1]:.2%}")

# Tab 2: Model Training
with tab2:
    st.markdown('<p class="sub-header">🧠 Model Training & Evaluation</p>', unsafe_allow_html=True)
    
    train_button = st.button("🚀 Train Model", use_container_width=True)
    
    if train_button:
        # Show loading spinner
        with st.spinner("Training model... This may take a moment."):
            # Preprocess data
            X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(
                df, scaling_method, handle_imbalance, use_pca, n_components if use_pca else None
            )
            
            # Train model
            model = train_model(X_train, y_train, n_estimators, max_depth, min_samples_split)
            
            # Save model and data to session state
            st.session_state.model = model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.scaler = scaler
            st.session_state.feature_names = feature_names
            
            # Evaluate model
            evaluation = evaluate_model(model, X_test, y_test, st.session_state.threshold)
            
            # Success message
            st.success("Model trained successfully!")
        
        # If model is trained, show evaluation
        if st.session_state.model is not None:
            # Model metrics
            st.markdown("### 📊 Model Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Accuracy", f"{evaluate_model(st.session_state.model, st.session_state.X_test, st.session_state.y_test, st.session_state.threshold)['accuracy']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Precision", f"{evaluate_model(st.session_state.model, st.session_state.X_test, st.session_state.y_test, st.session_state.threshold)['precision']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Recall", f"{evaluate_model(st.session_state.model, st.session_state.X_test, st.session_state.y_test, st.session_state.threshold)['recall']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("F1 Score", f"{evaluate_model(st.session_state.model, st.session_state.X_test, st.session_state.y_test, st.session_state.threshold)['f1']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Confusion matrix
            st.markdown("### Confusion Matrix")
            evaluation = evaluate_model(st.session_state.model, st.session_state.X_test, st.session_state.y_test, st.session_state.threshold)
            cm_fig = plot_confusion_matrix(evaluation["confusion_matrix"], plot_style)
            
            if plot_style == "Plotly":
                st.plotly_chart(cm_fig)
            else:
                st.pyplot(cm_fig)
            
            # ROC and PR curves
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ROC Curve")
                roc_fig = plot_roc_curve(evaluation["fpr"], evaluation["tpr"], evaluation["roc_auc"], plot_style)
                
                if plot_style == "Plotly":
                    st.plotly_chart(roc_fig)
                else:
                    st.pyplot(roc_fig)
            
            with col2:
                st.markdown("### Precision-Recall Curve")
                pr_fig = plot_pr_curve(evaluation["recall_curve"], evaluation["precision_curve"], evaluation["pr_auc"], plot_style)
                
                if plot_style == "Plotly":
                    st.plotly_chart(pr_fig)
                else:
                    st.pyplot(pr_fig)
            
            # Threshold slider
            st.markdown('<div class="threshold-slider">', unsafe_allow_html=True)
            st.markdown("### Detection Threshold Adjustment")
            threshold = st.slider(
                "Adjust detection threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.threshold,
                step=0.05,
                help="Higher threshold = higher precision, lower recall"
            )
            
            # Update threshold if changed
            if threshold != st.session_state.threshold:
                st.session_state.threshold = threshold
                st.experimental_rerun()
            
            st.markdown("""
            **Note:** Adjusting the threshold allows you to balance between:
            - **Higher precision** (fewer false positives) - Move threshold up
            - **Higher recall** (fewer false negatives) - Move threshold down
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Threshold analysis
            st.markdown("### Threshold Analysis")
            threshold_fig = plot_threshold_analysis(st.session_state.y_test, evaluation["y_proba"], plot_style)
            
            if plot_style == "Plotly":
                st.plotly_chart(threshold_fig)
            else:
                st.pyplot(threshold_fig)
            
            # Feature importance
            st.markdown('<div class="feature-importance">', unsafe_allow_html=True)
            st.markdown("### Feature Importance")
            
            importance_fig = plot_feature_importance(st.session_state.model, st.session_state.feature_names, plot_style)
            
            if plot_style == "Plotly":
                st.plotly_chart(importance_fig)
            else:
                st.pyplot(importance_fig)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        if st.session_state.model is None:
            st.info("Click the 'Train Model' button to start training the model with the selected parameters.")

# Tab 3: Real-time Detection
with tab3:
    st.markdown('<p class="sub-header">🔍 Real-time Fraud Detection</p>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("Please train a model in the 'Model Training' tab first.")
    else:
        st.markdown("""
        Test the model with custom transactions or generate random ones to see how the model performs in real-time.
        """)
        
        # Transaction input method
        input_method = st.radio(
            "Choose input method:",
            ["Generate random transaction", "Custom transaction"]
        )
        
        if input_method == "Generate random transaction":
            col1, col2 = st.columns([1, 2])
            
            with col1:
                fraud_probability = st.slider(
                    "Fraud probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="Higher value = more likely to generate fraudulent transaction"
                )
                
                generate_button = st.button("🎲 Generate Random Transaction", use_container_width=True)
            
            with col2:
                st.markdown("""
                **About Random Generation:**
                
                The generator creates transactions with characteristics that may resemble:
                - Genuine transactions (normal amounts, typical patterns)
                - Fraudulent transactions (unusual amounts, suspicious patterns)
                
                The fraud probability slider adjusts how likely the generator will create a transaction with fraud-like characteristics.
                """)
            
            if generate_button:
                # Generate random transaction
                transaction, actual_fraud = generate_random_transaction(fraud_probability)
                
                # Prepare for prediction
                transaction_df = pd.DataFrame([transaction])
                
                # Scale features if scaler exists
                if st.session_state.scaler is not None:
                    transaction_scaled = st.session_state.scaler.transform(transaction_df)
                else:
                    transaction_scaled = transaction_df.values
                
                # Make prediction
                fraud_proba = st.session_state.model.predict_proba(transaction_scaled)[0, 1]
                is_fraud = fraud_proba >= st.session_state.threshold
                
                # Display result with animation
                st.markdown("### 🔍 Detection Result")
                
                # Animated progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(101):
                    progress_bar.progress(i)
                    status_text.text(f"Analyzing transaction... {i}%")
                    time.sleep(0.01)
                
                progress_bar.empty()
                status_text.empty()
                
                # Display result
                if is_fraud:
                    st.markdown(f'''
                    <div class="fraud-alert">
                        <h3>⚠️ Fraudulent Transaction Detected!</h3>
                        <p><strong>Fraud Probability:</strong> {fraud_proba:.2%}</p>
                        <p><strong>Transaction Amount:</strong> ${transaction["Amount"]:.2f}</p>
                        <p><strong>Actual Label (for demo):</strong> {"Fraud" if actual_fraud else "Genuine"}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="genuine-alert">
                        <h3>✅ Genuine Transaction</h3>
                        <p><strong>Fraud Probability:</strong> {fraud_proba:.2%}</p>
                        <p><strong>Transaction Amount:</strong> ${transaction["Amount"]:.2f}</p>
                        <p><strong>Actual Label (for demo):</strong> {"Fraud" if actual_fraud else "Genuine"}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Display transaction details
                st.markdown("### Transaction Details")
                
                # Format transaction for display
                display_transaction = {
                    "Amount": f"${transaction['Amount']:.2f}",
                    "Time": f"{transaction['Time'] / 3600:.1f} hours",
                }
                
                # Add top 5 important features
                if st.session_state.feature_names:
                    importances = st.session_state.model.feature_importances_
                    indices = np.argsort(importances)[::-1][:5]
                    
                    for idx in indices:
                        if idx < len(st.session_state.feature_names):
                            feature_name = st.session_state.feature_names[idx]
                            if feature_name in transaction:
                                display_transaction[feature_name] = f"{transaction[feature_name]:.4f}"
                
                st.json(display_transaction)
                
                # Feature contribution
                st.markdown("### Feature Contribution to Prediction")
                
                # Get feature importance
                importances = st.session_state.model.feature_importances_
                
                # Get top 10 features
                top_indices = np.argsort(importances)[::-1][:10]
                top_features = [st.session_state.feature_names[i] for i in top_indices if i < len(st.session_state.feature_names)]
                top_values = [transaction[f] if f in transaction else 0 for f in top_features]
                
                # Create contribution chart
                if plot_style == "Plotly":
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=top_features,
                        y=top_values,
                        marker=dict(
                            color=["red" if v < 0 else "green" for v in top_values],
                            opacity=0.7
                        )
                    ))
                    
                    fig.update_layout(
                        title="Top Feature Values for This Transaction",
                        xaxis_title="Feature",
                        yaxis_title="Value",
                        height=400
                    )
                    
                    st.plotly_chart(fig)
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(top_features, top_values)
                    
                    # Color bars based on value
                    for i, bar in enumerate(bars):
                        bar.set_color("red" if top_values[i] < 0 else "green")
                        bar.set_alpha(0.7)
                    
                    ax.set_xlabel("Feature")
                    ax.set_ylabel("Value")
                    ax.set_title("Top Feature Values for This Transaction")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig)
        
        else:  # Custom transaction
            st.markdown("### Enter Custom Transaction Details")
            
            # Create form for custom transaction
            with st.form("custom_transaction_form"):
                amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=25000.0, value=100.0)
                
                # Time input (convert to seconds)
                time_hours = st.slider("Time (hours)", min_value=0, max_value=24, value=12)
                time_seconds = time_hours * 3600
                
                # Add inputs for top 5 important features
                custom_features = {}
                
                if st.session_state.model is not None and st.session_state.feature_names:
                    importances = st.session_state.model.feature_importances_
                    indices = np.argsort(importances)[::-1][:5]
                    
                    for idx in indices:
                        if idx < len(st.session_state.feature_names):
                            feature_name = st.session_state.feature_names[idx]
                            if feature_name not in ["Time", "Amount"]:
                                custom_features[feature_name] = st.slider(
                                    f"{feature_name} value",
                                    min_value=-5.0,
                                    max_value=5.0,
                                    value=0.0,
                                    step=0.1
                                )
                
                submit_button = st.form_submit_button("🔍 Analyze Transaction")
            
            if submit_button:
                # Create transaction dictionary
                transaction = {f"V{i+1}": 0.0 for i in range(28)}  # Default values
                transaction["Time"] = time_seconds
                transaction["Amount"] = amount
                
                # Update with custom features
                for feature, value in custom_features.items():
                    transaction[feature] = value
                
                # Prepare for prediction
                transaction_df = pd.DataFrame([transaction])
                
                # Scale features if scaler exists
                if st.session_state.scaler is not None:
                    transaction_scaled = st.session_state.scaler.transform(transaction_df)
                else:
                    transaction_scaled = transaction_df.values
                
                # Make prediction
                fraud_proba = st.session_state.model.predict_proba(transaction_scaled)[0, 1]
                is_fraud = fraud_proba >= st.session_state.threshold
                
                # Display result with animation
                st.markdown("### 🔍 Detection Result")
                
                # Animated progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(101):
                    progress_bar.progress(i)
                    status_text.text(f"Analyzing transaction... {i}%")
                    time.sleep(0.01)
                
                progress_bar.empty()
                status_text.empty()
                
                # Display result
                if is_fraud:
                    st.markdown(f'''
                    <div class="fraud-alert">
                        <h3>⚠️ Fraudulent Transaction Detected!</h3>
                        <p><strong>Fraud Probability:</strong> {fraud_proba:.2%}</p>
                        <p><strong>Transaction Amount:</strong> ${amount:.2f}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="genuine-alert">
                        <h3>✅ Genuine Transaction</h3>
                        <p><strong>Fraud Probability:</strong> {fraud_proba:.2%}</p>
                        <p><strong>Transaction Amount:</strong> ${amount:.2f}</p>
                    </div>
                    ''', unsafe_allow_html=True)

# Tab 4: Performance Analysis
with tab4:
    st.markdown('<p class="sub-header">📈 Performance Analysis</p>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("Please train a model in the 'Model Training' tab first.")
    else:
        st.markdown("""
        Analyze model performance across different scenarios and transaction types.
        """)
        
        # Batch testing
        st.markdown("### 🧪 Batch Testing")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            batch_size = st.slider("Number of transactions", min_value=10, max_value=1000, value=100, step=10)
            fraud_ratio = st.slider("Fraud ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
            
            batch_test_button = st.button("🧪 Run Batch Test", use_container_width=True)
        
        with col2:
            st.markdown("""
            **Batch Testing:**
            
            This feature generates a batch of random transactions with the specified fraud ratio and tests the model's performance.
            
            Use this to:
            - Simulate real-world scenarios
            - Test model robustness
            - Analyze performance patterns
            """)
        
        if batch_test_button:
            # Show loading spinner
            with st.spinner(f"Generating and analyzing {batch_size} transactions..."):
                # Generate batch of transactions
                transactions = []
                actual_labels = []
                
                for _ in range(batch_size):
                    is_fraud = np.random.random() < fraud_ratio
                    transaction, _ = generate_random_transaction(0.9 if is_fraud else 0.1)
                    transactions.append(transaction)
                    actual_labels.append(1 if is_fraud else 0)
                
                # Prepare for prediction
                batch_df = pd.DataFrame(transactions)
                
                # Scale features if scaler exists
                if st.session_state.scaler is not None:
                    batch_scaled = st.session_state.scaler.transform(batch_df)
                else:
                    batch_scaled = batch_df.values
                
                # Make predictions
                fraud_probas = st.session_state.model.predict_proba(batch_scaled)[:, 1]
                predicted_labels = (fraud_probas >= st.session_state.threshold).astype(int)
                
                # Calculate metrics
                accuracy = accuracy_score(actual_labels, predicted_labels)
                precision = precision_score(actual_labels, predicted_labels)
                recall = recall_score(actual_labels, predicted_labels)
                f1 = f1_score(actual_labels, predicted_labels)
                
                # Confusion matrix
                cm = confusion_matrix(actual_labels, predicted_labels)
            
            # Display results
            st.markdown("### Batch Test Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Precision", f"{precision:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Recall", f"{recall:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("F1 Score", f"{f1:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Confusion matrix
            st.markdown("### Confusion Matrix")
            cm_fig = plot_confusion_matrix(cm, plot_style)
            
            if plot_style == "Plotly":
                st.plotly_chart(cm_fig)
            else:
                st.pyplot(cm_fig)
            
            # Prediction distribution
            st.markdown("### Prediction Distribution")
            
            if plot_style == "Plotly":
                fig = px.histogram(
                    x=fraud_probas,
                    color=actual_labels,
                    nbins=50,
                    labels={"x": "Fraud Probability", "color": "Actual Class"},
                    color_discrete_map={0: "#4CAF50", 1: "#F44336"},
                    title="Distribution of Fraud Probabilities"
                )
                
                # Add threshold line
                fig.add_vline(
                    x=st.session_state.threshold,
                    line_dash="dash",
                    line_color="black",
                    annotation_text=f"Threshold: {st.session_state.threshold}"
                )
                
                st.plotly_chart(fig)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for label, color in zip([0, 1], ["#4CAF50", "#F44336"]):
                    mask = np.array(actual_labels) == label
                    ax.hist(
                        np.array(fraud_probas)[mask],
                        bins=50,
                        alpha=0.7,
                        color=color,
                        label=f"{'Genuine' if label == 0 else 'Fraud'}"
                    )
                
                ax.axvline(
                    st.session_state.threshold,
                    color="black",
                    linestyle="--",
                    label=f"Threshold: {st.session_state.threshold}"
                )
                
                ax.set_xlabel("Fraud Probability")
                ax.set_ylabel("Count")
                ax.set_title("Distribution of Fraud Probabilities")
                ax.legend()
                
                st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <p>Made with ❤️ by Sai Rupa Jhade | Data Science Portfolio</p>
        <p>Last updated: March 2024</p>
    </div>
    """,
    unsafe_allow_html=True
)

