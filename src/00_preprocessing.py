import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Load the dataset
DATA_PATH = r'C:\Users\Chris Jones\Desktop\Python Work\Credit Card Fraud Detection\src\data\raw\creditcard.csv'
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'processed'))
PROCESSED_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'features.npy')
PROCESSED_LABELS_PATH = os.path.join(PROCESSED_DIR, 'labels.npy')

df = pd.read_csv(DATA_PATH)

# Basic info
def print_basic_info(df):
    print('Shape:', df.shape)
    print('Columns:', df.columns.tolist())
    print('Missing values:', df.isnull().sum().sum())
    print('Class distribution:')
    print(df['Class'].value_counts())

# Preprocessing: scale features except 'Class'
def preprocess_data(df):
    features = df.drop('Class', axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, df['Class']

if __name__ == '__main__':
    print_basic_info(df)
    X, y = preprocess_data(df)
    print('Features shape:', X.shape)
    print('Labels shape:', y.shape)
    # Save processed data
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(PROCESSED_FEATURES_PATH, X)
    np.save(PROCESSED_LABELS_PATH, y)
    print(f'Processed features saved to {PROCESSED_FEATURES_PATH}')
    print(f'Processed labels saved to {PROCESSED_LABELS_PATH}')

