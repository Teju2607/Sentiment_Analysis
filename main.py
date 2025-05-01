""" Train and test classification model """
import os
import pickle
import gzip
import numpy as np
from sklearn import naive_bayes
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import urllib.request
import zipfile


def main():
    """ Train and test a model """

    col_names = ["polarity", "id", "date", "query", "user", "text"]

    data_dir = "data"
    train_data_file = 'data/trainingandtestdata/training.1600000.processed.noemoticon.csv'
    test_data_file = 'data/trainingandtestdata/testdata.manual.2009.06.14.csv'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not all([os.path.isfile(fp) for fp in [train_data_file, test_data_file]]):
        print("Dataset not found, downloading it...")
        download_dataset()

    model_filename = "data/model.dat.gz"

    if not os.path.isfile(model_filename):
        print("Model not found, training it...")
        train_dataset = pd.read_csv(
            train_data_file,
            names=col_names,
            encoding="latin-1"
        )
        model = train_model(train_dataset)
        print("Saving the model...")
        pickle.dump(model, gzip.open(model_filename, "wb"))
    else:
        print("Loading the model...")
        model = pickle.load(gzip.open(model_filename, "rb"))

    test_dataset = pd.read_csv(
        test_data_file,
        names=col_names,
        encoding="latin-1"
    )

    test_dataset = test_dataset[test_dataset.polarity != 2]

    test_model(model, test_dataset)


def train_vectorizer(corpus, max_features=10000):
    """ Train the vectorizer """
    print("Training the vectorizer...")
    vectorizer = CountVectorizer(decode_error='ignore', max_features=max_features)
    vectorizer.fit(corpus)
    print("Done.")
    return vectorizer


def extract_features(vectorizer, text):
    """ Extract text features """
    return vectorizer.transform(text)


def train_model(dataset):
    """ Train a new model """
    text_train = dataset.text
    vectorizer = train_vectorizer(text_train)
    vectorizer.stop_words_ = set({})
    print("Extracting features...")
    x_train = extract_features(vectorizer, text_train)
    y_train = dataset.polarity
    model = naive_bayes.MultinomialNB()
    print("Training the model...")
    model.fit(x_train, y_train)
    model.vectorizer = vectorizer
    return model


def test_model(model, dataset):
    """ Test the given model (confusion matrix) """
    print("Testing the model...")
    text_test = dataset.text
    x_test = extract_features(model.vectorizer, text_test)
    y_test = dataset.polarity
    y_predicted = model.predict(x_test)
    cmat = confusion_matrix(y_test, y_predicted)
    print(np.around(cmat / cmat.astype(np.float64).sum(axis=1)[:, np.newaxis] * 100))
    print("Accuracy: %.3f" % (float(np.trace(cmat)) / float(np.sum(cmat))))


def download_dataset():
    """ Fetch sentiment analysis dataset (Stanford website) """
    url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    file_name = 'data/trainingandtestdata.zip'

    print(f"Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, file_name)
    print("Download complete.")

    with zipfile.ZipFile(file_name, "r") as zin:
        folder = 'data/trainingandtestdata/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        zin.extractall(folder)
    print("Extraction complete.")


if __name__ == '__main__':
    main()
