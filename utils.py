def is_float_column(df, feature):
    if isinstance(df[feature][0], float):
        return True
    return False


def index_not_float(df, add_house=True):
    features = []
    for i in range(0, len(df.columns)):
        if df.columns[i] == "Hogwarts House" and not add_house:
            continue
        if not is_float_column(df, df.columns[i]):
            features.append(i)
    return features
