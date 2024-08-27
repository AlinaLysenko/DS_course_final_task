"""
Script loads the latest trained PyTorch model, data for inference, and predicts results.
Imports necessary packages and modules.
"""

import json
import logging
import os
import sys
import pandas as pd
from datetime import datetime
from typing import List

from joblib import load
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Use an environment variable or default to 'settings.json'
CONF_FILE = "settings.json"  # if you have problems with env variables
# CONF_FILE = os.getenv('CONF_PATH')


# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, (conf['general']['data_dir'])))
RAW_DATA_DIR = os.path.abspath(os.path.join(DATA_DIR, (conf['general']['raw_data_subdir'])))
PROCESSED_DATA_DIR = os.path.abspath(os.path.join(DATA_DIR, (conf['general']['processed_data_subdir'])))
RAW_DATA_DIR = os.path.abspath(os.path.join(DATA_DIR, (conf['general']['raw_data_subdir'])))
PROCESSED_DATA_DIR = os.path.abspath(os.path.join(DATA_DIR, (conf['general']['processed_data_subdir'])))
MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, (conf['general']['models_dir'])))
vector_data = conf['general']['vector_data']
models = conf['general']['models_subdirs']
RESULTS_DIR = os.path.abspath(os.path.join(ROOT_DIR, (conf['general']['results_dir'])))
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def load_data():
    logger.info("Start loading data.")
    vector_data_content = {
        key: load_npz(os.path.join(PROCESSED_DATA_DIR, f"{path}_test.npz"))
        for key, path in vector_data.items()
    }
    logger.info("Processed data is loaded.")
    return vector_data_content


def get_model_path(model_name, data_key) -> str:
    """Gets the path of the latest saved model"""
    return os.path.abspath(os.path.join(os.path.join(MODEL_DIR, model_name), f'{data_key}_model.pkl'))


def evaluate_model(predictions, target, model_name, data_key):
    accuracy = accuracy_score(predictions, target)
    precision = precision_score(predictions, target)
    recall = recall_score(predictions, target)
    f1 = f1_score(predictions, target)
    return {
        'Model': model_name,
        'Vectorization_Type': data_key.split('_')[0],
        'Normalization_Type': data_key.split('_')[1],
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1
    }


def save_results(prediction_table, metrics_table):
    path = datetime.now().strftime(conf['general']['datetime_format'])
    metrics_path = os.path.join(RESULTS_DIR, path + '_metrics.csv')
    predictions_path = os.path.join(RESULTS_DIR, path + '_predictions.csv')
    prediction_table.to_csv(predictions_path, index=False)
    logging.info(f'Predictions saved to {predictions_path}')
    pd.DataFrame(metrics_table, columns=['Model', 'Vectorization_Type', 'Normalization_Type', 'Accuracy', 'Precision', 'Recall', 'F1_Score']).to_csv(metrics_path, index=False)
    logging.info(f'Metrics saved to {metrics_path}')


def predict_and_evaluate_models():
    column_names = ['review', 'sentiment']
    data_train = pd.read_csv(f'{RAW_DATA_DIR}/test.csv', names=column_names)
    target = data_train['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    vector_data_content = load_data()
    results = []
    for model_code, model_name in models.items():
        for data_key, data_value in vector_data_content.items():
            logger.info(f'Start predicting using {model_name} with {data_key}')
            model = load(get_model_path(model_name, data_key))
            predictions = model.predict(data_value)
            results.append(evaluate_model(predictions, target, model_name, data_key))
            data_train[f'{model_code}_{data_key}_predictions'] = predictions
            data_train[f'{model_code}_{data_key}_predictions'] = data_train[f'{model_code}_{data_key}_predictions'].apply(lambda x: 'positive' if x == 1 else 'negative')
    save_results(data_train, results)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Starting script...")
    predict_and_evaluate_models()
    logging.info('Inference completed successfully.')


if __name__ == "__main__":
    main()
