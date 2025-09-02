import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Card Fraud Detection Dashboard", layout="wide")

# Load sample data and models
def load_data():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'sample'))
    X = np.load(os.path.join(data_dir, 'features_sample.npy'))
    y = np.load(os.path.join(data_dir, 'labels_sample.npy'))
    return X, y

def load_models():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
    models = {
        'Isolation Forest': joblib.load(os.path.join(models_dir, 'isolation_forest.joblib')),
        'One-Class SVM': joblib.load(os.path.join(models_dir, 'one_class_svm.joblib')),
        'Local Outlier Factor': joblib.load(os.path.join(models_dir, 'local_outlier_factor.joblib')),
        'Elliptic Envelope': joblib.load(os.path.join(models_dir, 'elliptic_envelope.joblib')),
        'Random Forest': joblib.load(os.path.join(models_dir, 'random_forest.joblib')),
        'MLP Classifier': joblib.load(os.path.join(models_dir, 'mlp_classifier.joblib')),
    }
    return models

X, y = load_data()
models = load_models()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

st.sidebar.title("About")
st.sidebar.info("""
**Credit Card Fraud Detection**

- Compare multiple anomaly and supervised models
- Visualize confusion matrix, ROC, and metrics
- Upload your own data for predictions
""")

st.title("Credit Card Fraud Detection Dashboard (Sample Data)")
st.write("Select a model to view its performance on the test set.")

model_name = st.selectbox("Choose a model:", list(models.keys()))
model = models[model_name]

def get_preds_and_probs(model, X):
    if hasattr(model, 'predict_proba'):
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
    elif hasattr(model, 'decision_function'):
        preds = model.predict(X)
        if model_name in ['Isolation Forest', 'One-Class SVM', 'Local Outlier Factor', 'Elliptic Envelope']:
            preds = np.where(preds == -1, 1, 0)
        scores = model.decision_function(X)
        probs = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        preds = model.predict(X)
        if model_name in ['Isolation Forest', 'One-Class SVM', 'Local Outlier Factor', 'Elliptic Envelope']:
            preds = np.where(preds == -1, 1, 0)
        probs = None
    return preds, probs

preds, probs = get_preds_and_probs(model, X_test)

precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
auc_score = None
if probs is not None:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_score = auc(fpr, tpr)

st.markdown("### Model Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precision", f"{precision:.3f}")
col2.metric("Recall", f"{recall:.3f}")
col3.metric("F1-score", f"{f1:.3f}")
col4.metric("AUC", f"{auc_score:.3f}" if auc_score is not None else "N/A")

if probs is not None:
    threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.01)
    preds = (probs >= threshold).astype(int)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    st.info(f"Metrics at threshold {threshold:.2f}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

if model_name == 'Random Forest' and hasattr(model, 'feature_importances_'):
    st.subheader("Feature Importances (Random Forest)")
    importances = model.feature_importances_
    fig_imp, ax_imp = plt.subplots(figsize=(10, 4))
    ax_imp.bar(range(len(importances)), importances)
    ax_imp.set_title("Feature Importances")
    st.pyplot(fig_imp)
elif model_name == 'MLP Classifier' and hasattr(model, 'coefs_'):
    st.subheader("Feature Weights (MLP)")
    fig_mlp, ax_mlp = plt.subplots(figsize=(10, 4))
    ax_mlp.bar(range(len(model.coefs_[0])), model.coefs_[0][:, 0])
    ax_mlp.set_title("First Layer Weights")
    st.pyplot(fig_mlp)

cm = confusion_matrix(y_test, preds)
cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
labels = np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(row, prow)] for row, prow in zip(cm, cm_percent)])
st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=labels, fmt="", cmap="rocket", cbar=True, linewidths=1, linecolor='white', ax=ax_cm)
ax_cm.set_xlabel("Predicted: 0=Not Fraud, 1=Fraud")
ax_cm.set_ylabel("Actual: 0=Not Fraud, 1=Fraud")
ax_cm.set_title("Confusion Matrix with Percentages")
st.pyplot(fig_cm)

st.subheader("Classification Report")
report = classification_report(y_test, preds, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

if probs is not None:
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend()
    st.pyplot(fig_roc)
else:
    st.info("ROC curve not available for this model.")

st.subheader("Misclassified Transactions (Sample)")
mis_idx = np.where(y_test != preds)[0]
if len(mis_idx) > 0:
    mis_df = pd.DataFrame(X_test[mis_idx])
    mis_df['Actual'] = y_test[mis_idx]
    mis_df['Predicted'] = preds[mis_idx]
    st.dataframe(mis_df.head(10))
else:
    st.success("No misclassified samples!")

st.write("---")
st.write("Upload a CSV file with the same features to get predictions:")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    user_X = user_df.values
    user_preds, _ = get_preds_and_probs(model, user_X)
    user_df['Prediction'] = user_preds
    st.write(user_df)
    st.success("Predictions complete. 1 = Fraud, 0 = Not Fraud")
