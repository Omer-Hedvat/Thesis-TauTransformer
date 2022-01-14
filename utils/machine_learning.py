def min_max_scaler(df1, features, df2=None, return_as_df=True):
    """
    activates min_max_scaler over a df and returns the normalized DataFrame
    :param df1: pandas DataFrame which we want to fit_transfomr on
    :param features: a list of columns which are the features
    :param df2: pandas DataFrame - an optional dataframe which we transform only
    :param return_as_df: a boolean flag which determines if we want Numpy array or Pandas DataFrame
    :return: normalized dataframe/s (features only)
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df1_norm = scaler.fit_transform(df1[features])
    if return_as_df:
        df1_norm = pd.DataFrame(df1_norm, columns=features)
    if df2 is not None:
        df2_norm = scaler.transform(df2[features])
        if return_as_df:
            df2_norm = pd.DataFrame(df2_norm, columns=features)
        return df1_norm, df2_norm
    return df1_norm

