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


def wasserstein_dist(X_arr, y_arr, feature_idx, cls_feature1, cls_feature2):
    from scipy.stats import wasserstein_distance
    dist = wasserstein_distance(X_arr[y_arr == cls_feature1, feature_idx], X_arr[y_arr == cls_feature2, feature_idx])
    return dist


def bhattacharyya_dist(X_arr, y_arr, feature_idx, cls_feature1, cls_feature2):
    dist = bhattacharyya_distance(X_arr[y_arr == cls_feature1, feature_idx], X_arr[y_arr == cls_feature2, feature_idx])
    return dist


def hellinger_dist(X_arr, y_arr, feature_idx, cls_feature1, cls_feature2):
    dist = hellinger(X_arr[y_arr == cls_feature1, feature_idx], X_arr[y_arr == cls_feature2, feature_idx])
    return dist


def jm_dist(X_arr, y_arr, feature_idx, cls_feature1, cls_feature2):
    dist = jm_distance(X_arr[y_arr == cls_feature1, feature_idx], X_arr[y_arr == cls_feature2, feature_idx])
    return dist


def norm_by_dist_type(feature_mat):
    from utils.general import calc_mean_std
    mean, std = calc_mean_std(feature_mat)
    norm_feature_mat = (feature_mat - mean) / std
    return norm_feature_mat


def execute_distance_func(X_arr, y_arr, dist_func_name, feature_idx, cls_feature1, cls_feature2):
    """
    Executes various distance functions by 'function_name' argument.
    The function calculates the distance between 2 vectors (df column), the vectors are values from the same column but w. different label values.
    by each function_name this function knows to call the right distance function
    :param df: Pandas DataFrame
    :param function_name: the name of the function
    :param feature: the name of the feature/column we want to use the distance on
    :param label1: value of label # 1
    :param label2: value of label # 2
    :return: distance value between the vectors
    """
    assert dist_func_name in ['wasserstein', 'bhattacharyya', 'jm', 'hellinger']
    return {
        'wasserstein': lambda: wasserstein_dist(X_arr, y_arr, feature_idx, cls_feature1, cls_feature2),
        'bhattacharyya': lambda: bhattacharyya_dist(X_arr, y_arr, feature_idx, cls_feature1, cls_feature2),
        'hellinger': lambda: hellinger_dist(X_arr, y_arr, feature_idx, cls_feature1, cls_feature2),
        'jm': lambda: jm_dist(X_arr, y_arr, feature_idx, cls_feature1, cls_feature2)
    }[dist_func_name]()


def calc_dist(X, y, all_features, dist_func_name):
    import numpy as np
    from utils.general import flatten
    from utils.machine_learning import min_max_scaler
    """
    Calculates distances of each feature w/ itself in different target classses
    for each DataFrame & distance functions
    :param dist_func_name: Distance function name
    return: df_dists, dist_dict
    df_dists - a flattened df of all features (each feature is a row)
    dist_dict - a dictionary of feature names & dataframes (e.g. {'feature_1': feature_1_df, ...}
    """
    X_tr_norm = min_max_scaler(X, all_features)
    distances = []
    classes = np.unique(y)
    for feature_idx in range(len(all_features)):
        class_dist = []
        for idx in range(len(classes)):
            cls_feature1 = classes[idx]
            class_row = [
                execute_distance_func(X_tr_norm, y, dist_func_name, feature_idx, cls_feature1, cls_feature2)
                if cls_feature1 != cls_feature2 else 0
                for cls_feature2 in classes[idx + 1:]
            ]
            class_dist.append(class_row)
        distances.append(class_dist)

    dists_dict = dict()
    two_dim_matrix = [flatten(distances[idx]) for idx in range(len(distances))]
    dists_arr = np.array([np.array(row) for row in two_dim_matrix])
    # dist_dict = {f'feature_{idx + 1}': pd.DataFrame(mat) for idx, mat in enumerate(distances)}
    dists_dict[dist_func_name] = dists_arr
    return dists_dict

