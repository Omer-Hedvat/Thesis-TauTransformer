import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def min_max_scaler(df, features):
    scaler = MinMaxScaler()
    numpy_norm = scaler.fit_transform(df[features])
    df_norm = pd.DataFrame(numpy_norm, columns=features)
    df_norm['label'] = df['label'].values
    return df_norm


def flatten(t):
    """

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
    # Calculate the square of the difference of ith distr elements
    list_of_squares = [((sqrt(p_i) - sqrt(q_i)) ** 2) for p_i, q_i in zip(p, q)]
    sosq = sum(list_of_squares)
    return sosq / sqrt(2)


