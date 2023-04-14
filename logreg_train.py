import numpy as np
import pandas as pd
import argparse
from logistic_regression import LogisticRegression
from data_analysis.utils import index_not_float, standardize, impute_by_target
from sklearn.metrics import accuracy_score
import threading

default_file_path = "./data/dataset_train.csv"
default_test_path = "./data/dataset_test.csv"
default_weights_path = "./data/thetas.csv"


def train_house(house, features, target, models):
    target_house = np.where(target == house, 1, 0)

    lr = LogisticRegression(house)
    lr.fit(features, target_house)
    lr.save(default_weights_path, house)

    models.append(lr)


def logreg(dataframe):
    houses = set(dataframe["Hogwarts House"].values)

    target = dataframe["Hogwarts House"].values
    target = target.reshape(-1, 1)

    to_remove = index_not_float(dataframe)
    features = dataframe.drop(dataframe.columns[to_remove], axis=1)
    features = features.drop("Potions", axis=1)
    features = features.drop("Arithmancy", axis=1)
    features = features.drop("Care of Magical Creatures", axis=1)
    features = features.values

    features = impute_by_target(features, dataframe["Hogwarts House"])
    features = standardize(features)

    with open(default_weights_path, "w") as f:
        f.write("Hogwarts House,Theta\n")
        f.close()

    print("Starting training...")

    models = []
    threads = []
    for house in houses:
        t = threading.Thread(target=train_house, args=(house, features, target, models))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for model in models:
        model.stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", help="File path to the dataset", default=default_file_path
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file, index_col=0)

        logreg(df)
    except FileNotFoundError:
        exit("Invalid file")
