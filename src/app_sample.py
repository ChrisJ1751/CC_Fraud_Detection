import numpy as np
import os

# Use the sample data for demo deployments
SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'sample'))
features_path = os.path.join(SAMPLE_DIR, 'features_sample.npy')
labels_path = os.path.join(SAMPLE_DIR, 'labels_sample.npy')

X = np.load(features_path)
y = np.load(labels_path)

# ...rest of your Streamlit code (unchanged)...
# Replace all previous X, y loading with the above
