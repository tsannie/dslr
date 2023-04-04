import sys
import pandas as pd
import numpy as np

file_path = "./data/dataset_train.csv"

def histogram(df):

    df = df.drop(df.columns[1:5], axis=1)

    courses = df.columns[1:]
    houses = set(df["Hogwarts House"])

    print(courses)
    print(houses)





if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: python histogram.py")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path, index_col="Index")

        histogram(df)
    except FileNotFoundError:
        print("Invalid file")
