"""
This script prepares the data, runs the training, and saves the model.
"""
import logging
import os
import json

from joblib import dump
from scipy.sparse import load_npz
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CONF_FILE = "settings.json" #if you have problems with env variables
#CONF_FILE = os.getenv('CONF_PATH')

logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

logger.info("Defining paths...")

# Configuration
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, (conf['general']['data_dir'])))
RAW_DATA_DIR = os.path.abspath(os.path.join(DATA_DIR, (conf['general']['raw_data_subdir'])))
PROCESSED_DATA_DIR = os.path.abspath(os.path.join(DATA_DIR, (conf['general']['processed_data_subdir'])))
MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, (conf['general']['models_dir'])))
vector_data = conf['general']['vector_data']
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    logger.info(f"Directory  {MODEL_DIR} was created.")

# Load data
def load_data():
    logger.info("Start loading data.")
    vector_data_content = {
        key: load_npz(os.path.join(PROCESSED_DATA_DIR, f"{path}_train.npz"))
        for key, path in vector_data.items()
    }
    logger.info("Processed data is loaded.")
    return vector_data_content


# Train model
def train_model():
    column_names = ['review', 'sentiment']
    data_train = pd.read_csv(f'{RAW_DATA_DIR}/train.csv', names=column_names)
    target = data_train['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    models = {
        (conf['general']['models_subdirs']['log_reg']): LogisticRegression(max_iter=1000, random_state=42),
        (conf['general']['models_subdirs']['svm']): LinearSVC(max_iter=5000, C=0.01, random_state=42),
        (conf['general']['models_subdirs']['rand_forest']): RandomForestClassifier(random_state=42)
    }
    vector_data_content = load_data()

    for model_name, model in models.items():
        logger.info(f'Start training models {model_name}')
        CURRENT_MODEL_DIR = os.path.abspath(os.path.join(MODEL_DIR, model_name))
        if not os.path.exists(CURRENT_MODEL_DIR):
            os.makedirs(CURRENT_MODEL_DIR)
            logger.info(f"Directory {CURRENT_MODEL_DIR} was created.")
        for data_key, data_value in vector_data_content.items():
            logger.info(f'Start training with {data_key} data')
            model.fit(data_value, target)
            dump(model, os.path.abspath(os.path.join(CURRENT_MODEL_DIR, f'{data_key}_model.pkl')))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Starting training process...")
    train_model()
    logger.info("Training process completed successfully.")
