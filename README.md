# Credit Card Fraud Detection

## Overview
This project implements a complete machine learning pipeline for detecting credit card fraud using anomaly detection and supervised learning models. It includes data preprocessing, exploratory data analysis (EDA), model selection, training, evaluation, visualization, and an interactive Streamlit dashboard.

## Project Structure
```
├── src/
│   ├── 00_preprocessing.py      # Data loading and preprocessing
│   ├── 01_EDA.py                # Exploratory Data Analysis
│   ├── 02_model_selection.py    # Model selection and setup
│   ├── 03_model_training.py     # Model training (with train/test split)
│   ├── 04_evaluation.py         # Model evaluation on test set
│   ├── 05_visualization.py      # Results visualization (ROC, confusion matrix)
│   ├── app.py                   # Streamlit dashboard (full data, not for deployment)
│   ├── app_sample.py            # Streamlit dashboard (sample data, for demo/deployment)
│   ├── data/                    # Data folders (raw, processed)
│   ├── models/                  # Saved models
├── requirements.txt
├── README.md
```

## Dataset
- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Description:** Contains transactions made by European cardholders in September 2013. Highly imbalanced: 492 frauds out of 284,807 transactions.
- **Note:** The raw dataset is not included in the repo due to size limits. Download it from Kaggle and place it in `src/data/raw/`.

## Steps

### 1. Data Preprocessing (`00_preprocessing.py`)
- Loads the raw CSV data.
- Scales features using `StandardScaler`.
- Saves processed features and labels as `.npy` files.

### 2. Exploratory Data Analysis (`01_EDA.py`)
- Visualizes class distribution (fraud vs. non-fraud).
- Plots feature distributions and correlation heatmap.

### 3. Model Selection (`02_model_selection.py`)
- Sets up a variety of models:
  - **Isolation Forest** (unsupervised anomaly detection)
  - **One-Class SVM** (unsupervised anomaly detection)
  - **Local Outlier Factor** (unsupervised anomaly detection)
  - **Elliptic Envelope** (unsupervised anomaly detection)
  - **Random Forest Classifier** (supervised)
  - **MLP Classifier** (supervised neural network)
- Models were chosen to compare classic anomaly detection with modern supervised approaches.

### 4. Model Training (`03_model_training.py`)
- Splits data into train/test sets (stratified, 70/30).
- Trains anomaly models on non-fraud training data only.
- Trains supervised models on the full training set.
- Saves all models to `src/models/`.

### 5. Model Evaluation (`04_evaluation.py`)
- Evaluates all models on the test set using:
  - Confusion matrix
  - Precision, recall, F1-score
  - Classification report
- **Key findings:**
  - Unsupervised models (Isolation Forest, One-Class SVM, LOF, Elliptic Envelope) struggle with precision and recall due to class imbalance.
  - Supervised models (Random Forest, MLP) perform much better, with Random Forest achieving the best balance of precision and recall.

### 6. Results Visualization (`05_visualization.py`)
- Plots ROC curves and confusion matrices for supervised models.
- Visualizes model performance for easy comparison.

### 7. Interactive Dashboard (`app.py` and `app_sample.py`)
- Built with Streamlit.
- `app.py`: For local use with full data (not suitable for deployment due to file size limits).
- `app_sample.py`: For demo/deployment, uses included small sample data.
- Allows users to:
  - Select and compare all models
  - View metrics, ROC, confusion matrix, and misclassified samples
  - Upload new data for prediction
  - Adjust decision threshold for probability-based models
  - See feature importances (Random Forest, MLP)

## Demo Deployment
- The repository includes a small, balanced sample dataset (`src/data/sample/`) and `app_sample.py` for easy deployment on Streamlit Cloud or similar platforms.
- For full analysis, run locally with the full dataset (see instructions above).

## Model Explanations

### Anomaly Detection Models
- **Isolation Forest:** Detects anomalies by isolating observations. Good for unsupervised outlier detection, but struggles with highly imbalanced data.
- **One-Class SVM:** Learns the boundary of normal data. High recall but low precision in this context.
- **Local Outlier Factor:** Measures local deviation of density. Performed poorly on this dataset.
- **Elliptic Envelope:** Assumes data is Gaussian and fits an ellipse. Not effective for this dataset's distribution.

### Supervised Models
- **Random Forest Classifier:** Ensemble of decision trees. Handles imbalance well, provides feature importances, and achieved the best results.
- **MLP Classifier:** Neural network. Performed well, but slightly less effective than Random Forest.

## Evaluation Summary
| Model                | Precision | Recall | F1-score | Notes                                      |
|----------------------|-----------|--------|----------|--------------------------------------------|
| Isolation Forest     | ~0.21     | ~0.23  | ~0.22    | Misses most frauds, many false positives   |
| One-Class SVM        | ~0.11     | ~0.79  | ~0.19    | High recall, very low precision            |
| Local Outlier Factor | ~0.01     | ~0.01  | ~0.01    | Extremely poor                            |
| Elliptic Envelope    | ~0.14     | ~0.17  | ~0.15    | Poor                                      |
| Random Forest        | ~0.96     | ~0.76  | ~0.85    | Best overall, strong real-world results    |
| MLP Classifier       | ~0.88     | ~0.71  | ~0.79    | Very good, but not as strong as RF         |

## How to Run
1. Clone the repo and download the dataset from Kaggle.
2. Place `creditcard.csv` in `src/data/raw/`.
3. Install requirements:
   ```
   pip install -r requirements.txt
   ```
4. Run preprocessing, training, and evaluation scripts as needed.
5. Launch the dashboard:
   ```
   cd src
   streamlit run app.py
   ```

## Notes
- Large files and datasets are excluded from the repo. See `.gitignore`.
- For best results, retrain models if you update the dataset.
- For production, consider using model versioning and secure data handling.

## License
MIT License

## Author
Chris Jones
