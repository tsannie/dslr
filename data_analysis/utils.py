import numpy as np


def colorize_house(house):
    if house == "Gryffindor":
        return "\033[31m" + house + "\033[0m"
    elif house == "Slytherin":
        return "\033[32m" + house + " \033[0m"
    elif house == "Ravenclaw":
        return "\033[36m" + house + " \033[0m"
    elif house == "Hufflepuff":
        return "\033[33m" + house + "\033[0m"


def colorize_plot(house):
    if house == "Gryffindor":
        return "red"
    elif house == "Slytherin":
        return "green"
    elif house == "Ravenclaw":
        return "blue"
    elif house == "Hufflepuff":
        return "orange"


def is_float_column(df, feature):
    if isinstance(df[feature][0], float) and not np.isnan(df[feature][0]):
        return True
    return False


def index_not_float(df, delete_house=True):
    features = []
    for i in range(0, len(df.columns)):
        if df.columns[i] == "Hogwarts House" and not delete_house:
            continue
        if not is_float_column(df, df.columns[i]):
            features.append(i)
    return features


def standardize(X):
    """Standardize the dataset X"""

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X


def impute(X):
    """Replace missing values with the mean of the column"""

    mean = np.nanmean(X, axis=0)
    missing = np.isnan(X)
    X[missing] = np.take(mean, np.where(missing)[1])
    return X


def impute_by_target(X, y):
    """Replace missing values with the mean of the column"""

    houses = set(y)
    for house in houses:
        house_indexes = np.where(y == house)
        house_features = X[house_indexes]
        mean = np.nanmean(house_features, axis=0)
        missing = np.isnan(house_features)
        house_features[missing] = np.take(mean, np.where(missing)[1])
        X[house_indexes] = house_features
    return X


def predict(X, theta):
    """Return the prediction of X using a logistic regression"""

    return 1 / (1 + np.exp(-np.dot(X, theta)))
