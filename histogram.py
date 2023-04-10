import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse
from utils import index_not_float, is_float_column

default_file_path = "./data/dataset_train.csv"


def histogram(df, courses):
    if courses:
        courses = courses.split(",")
        for course in courses:
            if course not in df.columns or not is_float_column(df, course):
                exit("Invalid course name.")
        courses.insert(0, "Hogwarts House")
        df = df[courses]
    else:
        to_remove = index_not_float(df, delete_house=False)
        df = df.drop(df.columns[to_remove], axis=1)
    courses = df.columns[1:]
    houses = set(df["Hogwarts House"])

    n_courses = len(courses)
    n_rows = math.ceil(math.sqrt(n_courses))
    n_cols = math.ceil(n_courses / n_rows)

    figure = plt.figure(figsize=(16, 12))
    figure.suptitle("Histograms of the {} courses".format(n_courses), fontsize=20)

    for i, course in enumerate(courses):
        ax = figure.add_subplot(n_rows, n_cols, i + 1)
        ax.set_title(course, fontsize=15)
        ax.set_xlabel("Score")
        ax.set_ylabel("Students")
        for house in houses:
            ax.hist(
                df[df["Hogwarts House"] == house][course],
                bins=10,
                alpha=0.5,
                label=house,
            )

    plt.legend(houses, loc="upper center", bbox_to_anchor=(1.5, 0.8))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", help="Path to the csv file", default=default_file_path
    )
    parser.add_argument(
        "-c", "--courses", help="List of courses to display separated by ','"
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file, index_col="Index")

        histogram(df, args.courses)
    except FileNotFoundError:
        print("Invalid file")
