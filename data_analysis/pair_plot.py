import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import index_not_float
import seaborn as sns

default_file_path = "../data/dataset_train.csv"


def pair_plot(df):
    to_remove = index_not_float(df, delete_house=False)
    df = df.drop(df.columns[to_remove], axis=1)
    courses = df.columns[1:]

    scatter_kws = {"s": 5, "alpha": 0.5}
    diag_kws = {"bins": 20, "alpha": 0.5}
    pairplot = sns.pairplot(
        df,
        hue="Hogwarts House",
        diag_kind="hist",
        plot_kws=scatter_kws,
        diag_kws=diag_kws,
        height=2,
    )

    for i, col in enumerate(pairplot.axes):
        for j, axes in enumerate(col):
            if i == 0:
                axes.set_title(
                    courses[j][:20] + "..." if len(courses[j]) > 20 else courses[j],
                    rotation=20,
                )
            if j == 0:
                axes.set_ylabel(
                    courses[i][:20] + "..." if len(courses[i]) > 20 else courses[i],
                    ha="right",
                    rotation=0,
                )
            axes.set_xlabel("")
            axes.set_xticks([])
            axes.set_yticks([])

    plt.subplots_adjust(
        top=0.90, bottom=0, left=0.1, right=0.93, hspace=0.3, wspace=0.3
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to the csv file")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_file, index_col=0)

        pair_plot(df)
    except FileNotFoundError:
        print("Invalid file")
