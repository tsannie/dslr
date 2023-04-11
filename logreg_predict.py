import argparse
import pandas as pd
import numpy as np
from utils import index_not_float, standardize, impute, predict

default_predictions_path = "./data/predictions.csv"


def display(most_probable, all_probabilities, idx):
    print("\033[34m{:3} \033[0m|".format(idx), end="")
    for proba in all_probabilities:
        if proba[0] == most_probable[0]:
            if proba[0] == "Gryffindor":
                print("\033[31m", end="")
            elif proba[0] == "Slytherin":
                print("\033[32m", end="")
            elif proba[0] == "Ravenclaw":
                print("\033[36m", end="")
            elif proba[0] == "Hufflepuff":
                print("\033[33m", end="")
            print(" {:3.0f}% \033[0m|".format(proba[1] * 100), end="")
        else:
            print(" {:3.0f}% |".format(proba[1] * 100), end="")
    print()


def logreg_predict(dataframe, thetas, show_predictions=False):
    houses = thetas.index

    thetas["Theta"] = thetas["Theta"].apply(
        lambda x: np.fromstring(x, dtype=float, sep=" ")
    )

    to_remove = index_not_float(dataframe)
    features = dataframe.drop(dataframe.columns[to_remove], axis=1)
    features = features.drop("Potions", axis=1)
    features = features.drop("Arithmancy", axis=1)
    features = features.drop("Care of Magical Creatures", axis=1)
    features = features.values

    features = impute(features)
    features = standardize(features)

    predictions = []
    if show_predictions:
        print("\033[33mIdx \033[0m|", end="")
        for house in houses:
            print("\033[34m{}\033[0m|".format(house[0:6]), end="")
        print()

    for i in range(0, len(features)):
        all_probabilities = []
        for house in houses:
            theta = thetas["Theta"][house]
            proba = predict(features[i], theta)
            all_probabilities.append((house, proba))

        most_probable = max(all_probabilities, key=lambda x: x[1])
        predictions.append((i, most_probable[0]))

        if show_predictions:
            display(most_probable, all_probabilities, i)

    return predictions


def save_predictions(predictions, path):
    with open(path, "w") as f:
        f.write("Index,Hogwarts House\n")
        for pred in predictions:
            f.write("{},{}\n".format(pred[0], pred[1]))
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset to predict", metavar="dataset_path")
    parser.add_argument(
        "thetas", help="Thetas to use for prediction", metavar="thetas_path"
    )
    parser.add_argument(
        "-s",
        "--show",
        help="Show predictions",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.dataset, index_col=0)
        thetas = pd.read_csv(args.thetas, index_col=0)

        predictions = logreg_predict(df, thetas, args.show)
        save_predictions(predictions, default_predictions_path)
    except FileNotFoundError:
        exit("Invalid file")
