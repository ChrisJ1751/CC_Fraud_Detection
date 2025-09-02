import numpy as np
import os
from sklearn.model_selection import train_test_split

# Use the processed data
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sample'))

features_path = os.path.join(PROCESSED_DIR, 'features.npy')
labels_path = os.path.join(PROCESSED_DIR, 'labels.npy')

sample_features_path = os.path.join(SAMPLE_DIR, 'features_sample.npy')
sample_labels_path = os.path.join(SAMPLE_DIR, 'labels_sample.npy')

# Load full data
X = np.load(features_path)
y = np.load(labels_path)

# Ensure at least 20 fraud and 980 non-fraud cases in the sample
fraud_idx = np.where(y == 1)[0]
nonfraud_idx = np.where(y == 0)[0]

n_fraud = min(20, len(fraud_idx))
# Ensure total sample size is 1000
n_nonfraud = 1000 - n_fraud

np.random.seed(42)
fraud_sample_idx = np.random.choice(fraud_idx, n_fraud, replace=False)
nonfraud_sample_idx = np.random.choice(nonfraud_idx, n_nonfraud, replace=False)

sample_idx = np.concatenate([fraud_sample_idx, nonfraud_sample_idx])
np.random.shuffle(sample_idx)

X_sample = X[sample_idx]
y_sample = y[sample_idx]

# Save sample
np.save(sample_features_path, X_sample)
np.save(sample_labels_path, y_sample)
print(f"Sample saved to {sample_features_path} and {sample_labels_path}")
