# Importing required libraries
import json
import logging
import os
import shutil
import string
import tempfile
import zipfile

import nltk
import pandas as pd
import requests
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = "settings.json"  # os.getenv('CONF_PATH')

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
TRAIN_URL = conf['data_load']['train_data_link']
TEST_URL = conf['data_load']['test_data_link']
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, (conf['general']['data_dir'])))
RAW_DATA_DIR = os.path.abspath(os.path.join(DATA_DIR, (conf['general']['raw_data_subdir'])))
PROCESSED_DATA_DIR = os.path.abspath(os.path.join(DATA_DIR, (conf['general']['processed_data_subdir'])))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(RAW_DATA_DIR):
    os.makedirs(RAW_DATA_DIR)
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

vectorizers = {
    'count_stemmed': CountVectorizer(min_df=5, ngram_range=(1, 2)),
    'count_lemmatized': CountVectorizer(min_df=5, ngram_range=(1, 2)),
    'tfidf_stemmed': TfidfVectorizer(min_df=5, ngram_range=(1, 2)),
    'tfidf_lemmatized': TfidfVectorizer(min_df=5, ngram_range=(1, 2))
}

common_words = set(conf['common_words'])

def find_csv_in_directory(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".csv"):
                return f'{os.path.join(root, file)}'


def handle_data(url):
    with (tempfile.TemporaryDirectory() as temp_dir):
        zip_path = os.path.join(temp_dir, 'data.zip')

        response = requests.get(url)
        with open(zip_path, 'wb') as zip_file:
            zip_file.write(response.content)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        csv_file_path = find_csv_in_directory(temp_dir)
        shutil.move(csv_file_path, RAW_DATA_DIR)


def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = unidecode(text)
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha() and token not in common_words]
    tokens = [token for token in tokens if token.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    stemmer = PorterStemmer()
    tokens_stem = [stemmer.stem(token) for token in tokens]

    lemmatizer = WordNetLemmatizer()
    tokens_lem = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens_stem), ' '.join(tokens_lem)


def get_path_for_vectorized_data(type, label):
    return os.path.join(PROCESSED_DATA_DIR, f'{conf["general"]["vector_data"][type]}_{label}.npz')


def vectorize_and_save_data(data, label):
    for key, vectorizer in vectorizers.items():
        data_key = key.split('_')[1]
        if label == 'train':
            transformed_data = vectorizer.fit_transform(data[f'processed_review_{data_key}'])
        elif label == 'test':
            transformed_data = vectorizer.transform(data[f'processed_review_{data_key}'])
        else:
            raise "No such type of data label"
        path = get_path_for_vectorized_data(key, label)
        save_npz(path, transformed_data)
        logger.info(f"Transformed {label} data using {key} vectorizer and saved to {path}.")


def load_and_process_data():
    logger.info("Loading and preparing Sentimental dataset...")
    handle_data(TRAIN_URL)
    handle_data(TEST_URL)
    logger.info(f'Data loaded and saved to {RAW_DATA_DIR}.')
    column_names = ['review', 'sentiment']

    logger.info("Start Processing data")
    data_train = pd.read_csv(os.path.join(RAW_DATA_DIR, 'train.csv'), names=column_names)
    data_train['processed_review_stemmed'], data_train['processed_review_lemmatized'] = zip(
        *data_train['review'].apply(preprocess_text))
    vectorize_and_save_data(data_train, "train")
    logger.info("Train data is processed and saved")

    data_test = pd.read_csv(os.path.join(RAW_DATA_DIR, 'test.csv'), names=column_names)
    data_test['processed_review_stemmed'], data_test['processed_review_lemmatized'] = zip(
        *data_test['review'].apply(preprocess_text))
    vectorize_and_save_data(data_test, "test")
    logger.info("Test data is processed and saved")


# Main execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Starting script...")
    load_and_process_data()
    logger.info("Script completed successfully.")
