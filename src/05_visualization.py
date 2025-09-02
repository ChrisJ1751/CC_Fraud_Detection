import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

# Load processed data
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'processed'))
FEATURES_PATH = os.path.join(PROCESSED_DIR, 'features.npy')
LABELS_PATH = os.path.join(PROCESSED_DIR, 'labels.npy')

X = np.load(FEATURES_PATH)
y = np.load(LABELS_PATH)

# Split data into train and test sets (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Load trained models
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
rf = joblib.load(os.path.join(MODELS_DIR, 'random_forest.joblib'))
mlp = joblib.load(os.path.join(MODELS_DIR, 'mlp_classifier.joblib'))

# Predict probabilities for ROC curve (supervised models)
rf_probs = rf.predict_proba(X_test)[:, 1]
mlp_probs = mlp.predict_proba(X_test)[:, 1]

# Predict classes for confusion matrix
rf_preds = rf.predict(X_test)
mlp_preds = mlp.predict(X_test)

# Plot ROC curves
def plot_roc(y_true, probs, label):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

plt.figure(figsize=(8, 6))
plot_roc(y_test, rf_probs, 'Random Forest')
plot_roc(y_test, mlp_probs, 'MLP Classifier')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.show()

# Plot confusion matrices
def plot_conf_matrix(y_true, y_pred, label):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix: {label}')
    plt.show()

plot_conf_matrix(y_test, rf_preds, 'Random Forest')
plot_conf_matrix(y_test, mlp_preds, 'MLP Classifier')
