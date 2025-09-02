import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import os
import joblib

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

# Train only on non-fraud data (Class == 0) for anomaly models
X_train_anomaly = X_train[y_train == 0]

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
os.makedirs(MODELS_DIR, exist_ok=True)

# Isolation Forest
iso_forest = IsolationForest(contamination=0.0017, random_state=42)
iso_forest.fit(X_train_anomaly)
joblib.dump(iso_forest, os.path.join(MODELS_DIR, 'isolation_forest.joblib'))
print('Isolation Forest trained and saved.')

# One-Class SVM
oc_svm = OneClassSVM(nu=0.0017, kernel='rbf', gamma='scale')
oc_svm.fit(X_train_anomaly)
joblib.dump(oc_svm, os.path.join(MODELS_DIR, 'one_class_svm.joblib'))
print('One-Class SVM trained and saved.')

# Local Outlier Factor (novelty=True for prediction on new data)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.0017, novelty=True)
lof.fit(X_train_anomaly)
joblib.dump(lof, os.path.join(MODELS_DIR, 'local_outlier_factor.joblib'))
print('Local Outlier Factor trained and saved.')

# Elliptic Envelope
elliptic = EllipticEnvelope(contamination=0.0017, random_state=42)
elliptic.fit(X_train_anomaly)
joblib.dump(elliptic, os.path.join(MODELS_DIR, 'elliptic_envelope.joblib'))
print('Elliptic Envelope trained and saved.')

# Random Forest Classifier (supervised, train on train set only)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest.joblib'))
print('Random Forest Classifier trained and saved.')

# MLP Classifier (supervised, train on train set only)
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
mlp.fit(X_train, y_train)
joblib.dump(mlp, os.path.join(MODELS_DIR, 'mlp_classifier.joblib'))
print('MLP Classifier trained and saved.')
