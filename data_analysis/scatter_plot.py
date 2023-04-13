import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import is_float_column, index_not_float

default_file_path = "../data/dataset_train.csv"


def scatter(df, c1, c2):
    if c1 and c2:
        courses = [c1, c2]
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
    fig, axes = plt.subplots(nrows=n_courses, ncols=n_courses, figsize=(16, 12))
    # set subtitle
    plt.suptitle("Scatter plot of the {} courses".format(n_courses), fontsize=20)

    for i, course1 in enumerate(courses):
        for j, course2 in enumerate(courses):
            ax = axes[i, j]

            if i == 0:
                ax.set_title(
                    course2[:20] + "..." if len(course2) > 20 else course2, rotation=20
                )
            if j == 0:
                ax.set_ylabel(
                    course1[:20] + "..." if len(course1) > 20 else course1,
                    ha="right",
                    rotation=0,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for house in houses:
                ax.scatter(
                    df[df["Hogwarts House"] == house][course1],
                    df[df["Hogwarts House"] == house][course2],
                    label=house,
                    s=2,
                )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to the csv file")
    parser.add_argument("-c1", "--course1", help="First course to compare")
    parser.add_argument("-c2", "--course2", help="Second course to compare")

    args = parser.parse_args()

    if args.course1 and not args.course2 or not args.course1 and args.course2:
        exit("You must provide two courses to compare")

    try:
        df = pd.read_csv(args.csv_file, index_col=0)

        scatter(df, args.course1, args.course2)
    except FileNotFoundError:
        print("Invalid file")
