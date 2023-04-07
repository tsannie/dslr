import argparse
import pandas as pd
import numpy as np
from utils import index_not_float, standardize, impute, predict

default_predictions_path = "./data/predictions.csv"

def logreg_predict(dataframe, thetas):
    houses = thetas.index

    thetas["Theta"] = thetas["Theta"].apply(
        lambda x: np.fromstring(x, dtype=float, sep=" ")
    )

    print(thetas["Theta"]["Ravenclaw"])

    to_remove = index_not_float(df)
    features = df.drop(df.columns[to_remove], axis=1)
    features = features.values

    features = impute(features)
    features = standardize(features)

    predictions = []

    for i in range(0, len(features)):
        print("\n\n------------------------------\nStudent: ", i)
        most_probable = (0, "")
        for house in houses:
            theta = thetas["Theta"][house]
            proba = predict(features[i], theta)
            print("Probability for {}: {:.0f}%".format(house, proba*100))
            if proba > most_probable[0]:
                most_probable = (proba, house)

        predictions.append((i, most_probable[1]))

    return predictions


def save_predictions(predictions, path):
    with open(path, "w") as f:
        f.write("Index,Hogwarts House\n")
        for pred in predictions:
            f.write("{},{}\n".format(pred[0], pred[1]))
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        help="Dataset to predict",
        metavar="dataset_path"
    )
    parser.add_argument(
        "thetas",
        help="Thetas to use for prediction",
        metavar="thetas_path"
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.dataset, index_col=0)
        thetas = pd.read_csv(args.thetas, index_col=0)

        predictions = logreg_predict(df, thetas)
        save_predictions(predictions, default_predictions_path)
    except FileNotFoundError:
        exit("Invalid file")
