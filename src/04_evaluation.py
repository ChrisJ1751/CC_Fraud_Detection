import numpy as np
import os
import joblib
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Load processed data
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'processed'))
FEATURES_PATH = os.path.join(PROCESSED_DIR, 'features.npy')
LABELS_PATH = os.path.join(PROCESSED_DIR, 'labels.npy')

X = np.load(FEATURES_PATH)
y = np.load(LABELS_PATH)

# Split data into train and test sets (stratified to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Load trained models
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
iso_forest = joblib.load(os.path.join(MODELS_DIR, 'isolation_forest.joblib'))
oc_svm = joblib.load(os.path.join(MODELS_DIR, 'one_class_svm.joblib'))
lof = joblib.load(os.path.join(MODELS_DIR, 'local_outlier_factor.joblib'))
elliptic = joblib.load(os.path.join(MODELS_DIR, 'elliptic_envelope.joblib'))
rf = joblib.load(os.path.join(MODELS_DIR, 'random_forest.joblib'))
mlp = joblib.load(os.path.join(MODELS_DIR, 'mlp_classifier.joblib'))

# Predict anomalies: Isolation Forest
iso_preds = iso_forest.predict(X_test)
iso_preds = np.where(iso_preds == -1, 1, 0)  # -1: anomaly, 1: fraud

# Predict anomalies: One-Class SVM
svm_preds = oc_svm.predict(X_test)
svm_preds = np.where(svm_preds == -1, 1, 0)

# Predict anomalies: Local Outlier Factor
lof_preds = lof.predict(X_test)
lof_preds = np.where(lof_preds == -1, 1, 0)

# Predict anomalies: Elliptic Envelope
elliptic_preds = elliptic.predict(X_test)
elliptic_preds = np.where(elliptic_preds == -1, 1, 0)

# Predict: Random Forest (supervised)
rf_preds = rf.predict(X_test)

# Predict: MLP Classifier (supervised)
mlp_preds = mlp.predict(X_test)

# Evaluation function
def evaluate(y_true, y_pred, model_name):
    print(f'\nEvaluation for {model_name}:')
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))
    print('Classification Report:')
    print(classification_report(y_true, y_pred, digits=4))
    print(f'Precision: {precision_score(y_true, y_pred):.4f}')
    print(f'Recall:    {recall_score(y_true, y_pred):.4f}')
    print(f'F1-score:  {f1_score(y_true, y_pred):.4f}')

def main():
    print(f'Test set size: {X_test.shape[0]} samples')
    evaluate(y_test, iso_preds, 'Isolation Forest')
    evaluate(y_test, svm_preds, 'One-Class SVM')
    evaluate(y_test, lof_preds, 'Local Outlier Factor')
    evaluate(y_test, elliptic_preds, 'Elliptic Envelope')
    evaluate(y_test, rf_preds, 'Random Forest Classifier')
    evaluate(y_test, mlp_preds, 'MLP Classifier')

if __name__ == '__main__':
    main()
