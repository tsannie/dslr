import numpy as np
import pandas as pd
import argparse
from logistic_regression import LogisticRegression
from utils import index_not_float, standardize, impute
from sklearn.metrics import accuracy_score
import os

default_file_path = "./data/dataset_train.csv"
default_test_path = "./data/dataset_test.csv"
default_weights_path = "./data/weights.csv"

def test_logreg(lr):
    df = pd.read_csv(default_test_path, index_col=0)
    to_remove = index_not_float(df)
    print(to_remove)
    features = df.drop(df.columns[to_remove], axis=1)
    features = features.values

    features = impute(features)
    features = standardize(features)

    print(features.shape)
    pred = lr.predict(features)

    df_truth = pd.read_csv("./data/dataset_truth.csv", index_col=0)
    truth = df_truth["Hogwarts House"].values
    truth = np.where(truth == "Gryffindor", 1, 0)

    print("Predictions: ", *pred, sep=" ")
    print("Truth: ", *truth, sep=" ")

    print("Accuracy: ", accuracy_score(truth, pred))


def logreg(dataframe):
    houses = set(dataframe["Hogwarts House"].values)

    target = dataframe["Hogwarts House"].values
    target = target.reshape(-1, 1)

    to_remove = index_not_float(dataframe)
    features = dataframe.drop(dataframe.columns[to_remove], axis=1)
    features = features.values

    # prepare features
    features = impute(features)
    features = standardize(features)

    for house in houses:
        print("Training for {}: ".format(house))

        target_house = np.where(target == house, 1, 0)

        lr = LogisticRegression()
        lr.fit(features, target_house)
        lr.save(default_weights_path)

    # test_logreg(lr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        help="File path to the dataset",
        default=default_file_path
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file, index_col=0)

        if os.path.exists(default_weights_path):
            os.remove(default_weights_path)

        logreg(df)
    except FileNotFoundError:
        exit("Invalid file")

