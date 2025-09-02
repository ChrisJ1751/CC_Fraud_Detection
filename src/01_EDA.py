import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load processed data
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'processed'))
FEATURES_PATH = os.path.join(PROCESSED_DIR, 'features.npy')
LABELS_PATH = os.path.join(PROCESSED_DIR, 'labels.npy')

X = np.load(FEATURES_PATH)
y = np.load(LABELS_PATH)

# Update: Use correct path for original data
ORIG_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'raw', 'creditcard.csv'))
df = pd.read_csv(ORIG_DATA_PATH)

# Basic stats
def print_class_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    print('Class distribution:')
    for label, count in zip(unique, counts):
        print(f'Class {int(label)}: {count}')

# Plot class distribution
def plot_class_distribution(y):
    sns.countplot(x=y)
    plt.title('Class Distribution (0: Non-Fraud, 1: Fraud)')
    plt.show()

# Plot feature distributions
def plot_feature_distributions(df):
    features = [col for col in df.columns if col not in ['Class']]
    df[features].hist(bins=50, figsize=(20, 15))
    plt.suptitle('Feature Distributions')
    plt.show()

# Plot correlation heatmap
def plot_correlation_heatmap(df):
    plt.figure(figsize=(16, 12))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.show()

def main():
    print_class_distribution(y)
    plot_class_distribution(y)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)

if __name__ == '__main__':
    main()
