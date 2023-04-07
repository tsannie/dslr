import numpy as np

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


def predict(X, theta):
    """Return the prediction of X using a logistic regression"""

    return 1 / (1 + np.exp(-np.dot(X, theta)))

