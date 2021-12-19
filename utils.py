import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def min_max_scaler(df, features):
    """
    activates min_max_scaler over a df and returns the normalized DataFrame
    :param df: pandas DataFrame
    :param features: a list of columns which are the features
    :return: normalized dataframe (features only)
    """
    scaler = MinMaxScaler()
    numpy_norm = scaler.fit_transform(df[features])
    df_norm = pd.DataFrame(numpy_norm, columns=features)
    df_norm['label'] = df['label'].values
    return df_norm


def flatten(t):
    """
    given a matrix, returns a flatten list
    :param t:
    :return:
    """
    return [item for sublist in t for item in sublist]


def JM_distance(p, q):
    b = bhattacharyya_distance(p, q)
    jm = 2 * (1 - np.exp(-b))
    return jm


def bhattacharyya_distance(p, q):
    mean_p, mean_q = p.mean(), q.mean()
    std_p = p.std() if p.std() != 0 else 0.00000000001
    std_q = q.std() if q.std() != 0 else 0.00000000001

    var_p, var_q = std_p ** 2, std_q ** 2
    b = (1 / 8) * ((mean_p - mean_q) ** 2) * (2 / (var_p + var_q)) + 0.5 * np.log((var_p + var_q) / (2 * (std_p * std_q)))
    return b


def hellinger(p, q):
    """Hellinger distance between two discrete distributions.
       Same as original version but without list comprehension
    """
    from math import sqrt
    # Calculate the square of the difference of ith distr elements
    list_of_squares = [((sqrt(p_i) - sqrt(q_i)) ** 2) for p_i, q_i in zip(p, q)]
    sosq = sum(list_of_squares)
    return sosq / sqrt(2)


def wasserstein_dist(df, feature, label1, label2):
    from scipy.stats import wasserstein_distance
    dist = wasserstein_distance(df.loc[df['label'] == label1, feature], df.loc[df['label']==label2, feature])
    return dist


def bhattacharyya_dist(df, feature, label1, label2):
    from utils import bhattacharyya_distance
    dist = bhattacharyya_distance(df.loc[df['label']==label1, feature], df.loc[df['label']==label2, feature])
    return dist


def hellinger_dist(df, feature, label1, label2):
    from utils import hellinger
    dist = hellinger(df.loc[df['label']==label1, feature], df.loc[df['label']==label2, feature])
    return dist


def jm_dist(df, feature, label1, label2):
    from utils import JM_distance
    dist = JM_distance(df.loc[df['label']==label1, feature], df.loc[df['label']==label2, feature])
    return dist




def norm_by_dist_type(feature_mat):
    mean, std = calc_mean_std(feature_mat)
    norm_feature_mat = (feature_mat - mean) / std
    return norm_feature_mat


def calc_mean_std(df):
    """
    Calculates matrix's mean & std (of entire matrix)
    :return: mean, std
    """
    mean = df.mean().mean()
    var = sum([((x - mean) ** 2) for x in flatten(df.values)]) / len(flatten(df.values))
    std = var ** 0.5
    return mean, std


def norm_by_dist_type(feature_mat):
    mean, std = calc_mean_std(feature_mat)
    norm_feature_mat = (feature_mat-mean)/std
    return norm_feature_mat


def calculateDistance(p1, p2):
    from math import sqrt
    dist = sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dist