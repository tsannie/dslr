import numpy as np
import pandas as pd
import argparse
from logistic_regression import LogisticRegression
from data_analysis.utils import index_not_float, standardize, impute_by_target
from sklearn.metrics import accuracy_score
import threading

default_weights_path = "./data/thetas.csv"


def train_house(house, features, target, models, batch_size, learning_rate, epochs):
    target_house = np.where(target == house, 1, 0)

    lr = LogisticRegression(house, learning_rate, batch_size, epochs)
    lr.fit(features, target_house)
    lr.save(default_weights_path, house)

    models.append(lr)


def logreg(dataframe, show_graphs=False, batch_size=16, learning_rate=0.01, epochs=10):
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

    print("Start of training...")

    models = []
    threads = []
    for house in houses:
        t = threading.Thread(
            target=train_house,
            args=(house, features, target, models, batch_size, learning_rate, epochs),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if show_graphs:
        for model in models:
            model.stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_file", help="CSV file to train on", metavar="csv_file_path"
    )
    parser.add_argument(
        "-w",
        "--weights",
        help="Path to save weights (default: {})".format(default_weights_path),
        metavar="weights_path",
        default=default_weights_path,
    )
    parser.add_argument(
        "-g",
        "--graph",
        help="Show graphs of training",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-b",
        "--batch",
        help="Batch size (default: 10) (0 for full batch)",
        default=10,
        type=int,
        metavar="batch_size",
    )
    parser.add_argument(
        "-l",
        "--learning",
        help="Learning rate (default: 0.01)",
        default=0.01,
        type=float,
        metavar="learning_rate",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="Number of epochs (default: 12)",
        default=12,
        type=int,
        metavar="epochs",
    )

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_file, index_col=0)
        if args.batch == 0:
            args.batch = df.shape[0]

        logreg(df, args.graph, args.batch, args.learning, args.epochs)
    except FileNotFoundError:
        exit("Invalid file")
