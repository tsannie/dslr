import argparse
import pandas as pd
from sklearn.metrics import accuracy_score


def compare_predictions(predictions, truth):
    accuracy = accuracy_score(truth, predictions)
    print(f"Accuracy: {accuracy}")

    correct = (predictions == truth).sum()
    print(f"Correct: {correct}")

    incorrect = (predictions != truth).sum()
    print(f"Incorrect: {incorrect}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predictions",
        help="Path to the predictions file",
        metavar="predictions_path",
    )
    parser.add_argument(
        "truth",
        help="Path to the truth file",
        metavar="truth_path",
    )
    args = parser.parse_args()

    try:
        predictions = pd.read_csv(args.predictions, index_col=0)
        truth = pd.read_csv(args.truth, index_col=0)

        compare_predictions(predictions, truth)
    except FileNotFoundError:
        exit("Invalid file")
