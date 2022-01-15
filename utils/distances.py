def jm_distance(p, q):
    import numpy as np

    b = bhattacharyya_distance(p, q)
    jm = 2 * (1 - np.exp(-b))
    return jm


def bhattacharyya_distance(p, q):
    import numpy as np

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
    dist = wasserstein_distance(df.loc[df['label'] == label1, feature], df.loc[df['label'] == label2, feature])
    return dist


def bhattacharyya_dist(df, feature, label1, label2):
    dist = bhattacharyya_distance(df.loc[df['label'] == label1, feature], df.loc[df['label'] == label2, feature])
    return dist


def hellinger_dist(df, feature, label1, label2):
    dist = hellinger(df.loc[df['label'] == label1, feature], df.loc[df['label'] == label2, feature])
    return dist


def jm_dist(df, feature, label1, label2):
    dist = jm_distance(df.loc[df['label'] == label1, feature], df.loc[df['label'] == label2, feature])
    return dist


def norm_by_dist_type(feature_mat):
    from utils.general import calc_mean_std
    mean, std = calc_mean_std(feature_mat)
    norm_feature_mat = (feature_mat - mean) / std
    return norm_feature_mat


def calculate_distance(p1, p2):
    from math import sqrt
    dist = sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dist
