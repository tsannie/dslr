import pandas as pd
import sys
from utils import index_not_float
import argparse

default_file_path = "./data/dataset_train.csv"


def ft_count(dataframe):
    """Count is the number of values in a set of data"""
    return dataframe.size


def ft_mean(dataframe):
    """Mean is the average of the numbers"""
    sum_data = 0
    for i in dataframe:
        sum_data += i
    return sum_data / ft_count(dataframe)


def ft_std(dataframe):
    """Standard deviation is a measure of the amount
    of variation or dispersion of a set of values"""
    n = ft_count(dataframe)
    mean = ft_mean(dataframe)
    sum_squared = 0
    for i in dataframe:
        sum_squared += (i - mean) ** 2
    return (sum_squared / (n - 1)) ** 0.5


def ft_percentile(dataframe, percentile):
    """Percentile is a measure used in statistics indicating the value
    below which a given percentage of observations in a group of observations falls"""
    n = ft_count(dataframe)
    i = (n - 1) * percentile / 100
    j = int(i)
    p = i - j

    if p == 0:
        return dataframe[j]
    else:
        return (1 - p) * dataframe[j] + p * dataframe[j + 1]


def ft_mad(dataframe):
    """Median Absolute Deviation is a more robust measure of dispersion"""
    n = ft_count(dataframe)
    mean = ft_mean(dataframe)
    sum_abs = 0
    for i in dataframe:
        sum_abs += abs(i - mean)
    return sum_abs / n


def analys(dataframe):
    return [
        ft_count(dataframe),
        ft_mean(dataframe),
        ft_std(dataframe),
        ft_percentile(dataframe, 0),
        ft_percentile(dataframe, 25),
        ft_percentile(dataframe, 50),
        ft_percentile(dataframe, 75),
        ft_percentile(dataframe, 100),
        ft_percentile(dataframe, 75) - ft_percentile(dataframe, 25),
        ft_mad(dataframe),
    ]


def ft_describe(dataframe):
    stats = [
        "",
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
        "iqr",
        "mad",
    ]
    describe = ""
    to_remove = index_not_float(dataframe)
    dataframe = dataframe.drop(dataframe.columns[to_remove], axis=1)
    features = dataframe.columns

    for stat in stats:
        if len(stat) > 12:
            stat = stat[:12] + "..."
        describe += "{:>15} |".format(stat)
    describe += "\n"

    for feature in features:
        clean_data = dataframe[feature].dropna()
        clean_data = clean_data.sort_values()
        clean_data = clean_data.reset_index(drop=True)

        if len(feature) > 12:
            feature = feature[:12] + "..."
        describe += "{:>15} |".format(feature)

        t = analys(clean_data)
        for i in t:
            if len(str(int(i))) > 10:
                describe += "{:>15.5e} |".format(i)
            else:
                describe += "{:>15.5f} |".format(i)
        describe += "\n"

    return describe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="file to describe (csv format)")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_file, index_col=0)

        print(ft_describe(df))
    except FileNotFoundError:
        print("Invalid file")
