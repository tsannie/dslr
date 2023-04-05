def is_float_column(df, feature):
    if isinstance(df[feature][0], float):
        return True
    return False
