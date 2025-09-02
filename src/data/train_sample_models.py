import numpy as np
import os
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.neural_network import MLPClassifier

# Load sample data
SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sample'))
features_path = os.path.join(SAMPLE_DIR, 'features_sample.npy')
labels_path = os.path.join(SAMPLE_DIR, 'labels_sample.npy')
X = np.load(features_path)
y = np.load(labels_path)

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
os.makedirs(MODELS_DIR, exist_ok=True)

# Train only on non-fraud for anomaly models
X_train_anomaly = X[y == 0]

# Isolation Forest
iso_forest = IsolationForest(contamination=0.02, random_state=42)
iso_forest.fit(X_train_anomaly)
joblib.dump(iso_forest, os.path.join(MODELS_DIR, 'isolation_forest_sample.joblib'))

# One-Class SVM
oc_svm = OneClassSVM(nu=0.02, kernel='rbf', gamma='scale')
oc_svm.fit(X_train_anomaly)
joblib.dump(oc_svm, os.path.join(MODELS_DIR, 'one_class_svm_sample.joblib'))

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=10, contamination=0.02, novelty=True)
lof.fit(X_train_anomaly)
joblib.dump(lof, os.path.join(MODELS_DIR, 'local_outlier_factor_sample.joblib'))

# Elliptic Envelope
elliptic = EllipticEnvelope(contamination=0.02, random_state=42)
elliptic.fit(X_train_anomaly)
joblib.dump(elliptic, os.path.join(MODELS_DIR, 'elliptic_envelope_sample.joblib'))

# Random Forest (supervised)
rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X, y)
joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest_sample.joblib'))

# MLP Classifier (supervised)
mlp = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=100, random_state=42)
mlp.fit(X, y)
joblib.dump(mlp, os.path.join(MODELS_DIR, 'mlp_classifier_sample.joblib'))

print("Sample models trained and saved.")
