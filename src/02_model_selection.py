import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import os

# Load processed data
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'processed'))
FEATURES_PATH = os.path.join(PROCESSED_DIR, 'features.npy')
LABELS_PATH = os.path.join(PROCESSED_DIR, 'labels.npy')

X = np.load(FEATURES_PATH)
y = np.load(LABELS_PATH)

# Model selection
MODELS = {
    'IsolationForest': IsolationForest(contamination=0.0017, random_state=42),
    'OneClassSVM': OneClassSVM(nu=0.0017, kernel='rbf', gamma='scale'),
    'LocalOutlierFactor': LocalOutlierFactor(n_neighbors=20, contamination=0.0017, novelty=True),
    'EllipticEnvelope': EllipticEnvelope(contamination=0.0017, random_state=42),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
    'MLPClassifier': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
}

def main():
    print('Available models:')
    for name in MODELS:
        print(f'- {name}')
    # Example: fit Isolation Forest
    print('\nFitting Isolation Forest...')
    model = MODELS['IsolationForest']
    model.fit(X)
    print('Isolation Forest fitted.')
    # Example: fit One-Class SVM
    print('\nFitting One-Class SVM...')
    model = MODELS['OneClassSVM']
    model.fit(X)
    print('One-Class SVM fitted.')
    # Example: fit Local Outlier Factor
    print('\nFitting Local Outlier Factor...')
    model = MODELS['LocalOutlierFactor']
    model.fit(X)
    print('Local Outlier Factor fitted.')
    # Example: fit Elliptic Envelope
    print('\nFitting Elliptic Envelope...')
    model = MODELS['EllipticEnvelope']
    model.fit(X)
    print('Elliptic Envelope fitted.')
    # Example: fit Random Forest Classifier
    print('\nFitting Random Forest Classifier...')
    model = MODELS['RandomForestClassifier']
    model.fit(X, y)
    print('Random Forest Classifier fitted.')
    # Example: fit MLP Classifier
    print('\nFitting MLP Classifier...')
    model = MODELS['MLPClassifier']
    model.fit(X, y)
    print('MLP Classifier fitted.')

if __name__ == '__main__':
    main()
