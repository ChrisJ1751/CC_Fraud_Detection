import numpy as np
import os

# Use the processed data
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'processed'))
if not os.path.exists(PROCESSED_DIR):
    PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sample'))

features_path = os.path.join(PROCESSED_DIR, 'features.npy')
labels_path = os.path.join(PROCESSED_DIR, 'labels.npy')

sample_features_path = os.path.join(SAMPLE_DIR, 'features_sample.npy')
sample_labels_path = os.path.join(SAMPLE_DIR, 'labels_sample.npy')

# Load full data
X = np.load(features_path)
y = np.load(labels_path)

# Take a small random sample (e.g., 1000 rows, stratified)
from sklearn.model_selection import train_test_split
_, X_sample, _, y_sample = train_test_split(X, y, test_size=1000, random_state=42, stratify=y)

# Save sample
np.save(sample_features_path, X_sample)
np.save(sample_labels_path, y_sample)
print(f"Sample saved to {sample_features_path} and {sample_labels_path}")
