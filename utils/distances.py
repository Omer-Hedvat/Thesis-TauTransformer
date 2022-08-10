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


def calculate_distance(p1, p2):
    from math import sqrt
    dist = sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return dist


def execute_distance_func(df, function_name, feature, label1, label2):
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
    assert function_name in ['wasserstein', 'bhattacharyya', 'jm', 'hellinger']
    return {
        'wasserstein': lambda: wasserstein_dist(df, feature, label1, label2),
        'bhattacharyya': lambda: bhattacharyya_dist(df, feature, label1, label2),
        'hellinger': lambda: hellinger_dist(df, feature, label1, label2),
        'jm': lambda: jm_dist(df, feature, label1, label2)
    }[function_name]()


def calc_dist(dist_func_name, X_tr, classes, label_column):
    """
    Calculates distances of each feature w/ itself in different target classses
    for each DataFrame & distance functions
    :param dist_func_name: Distance function name
    :param X_tr:
    :param classes: y_train
    return: df_dists, dist_dict
    df_dists - a flatten df of all features (each feature is a row)
    dist_dict - a dictionary of feature names & dataframes (e.g. {'feature_1': feature_1_df, ...}
    """
    import pandas as pd
    from utils.general import flatten

    features = X_tr.columns
    df = X_tr
    classes.reset_index(drop=True, inplace=True)
    df[label_column] = classes
    distances = []
    for feature in features:
        class_dist = []
        for cls_feature1 in classes.unique():
            class_row = [
                execute_distance_func(X_tr, dist_func_name, feature, cls_feature1, cls_feature2)
                if cls_feature1 != cls_feature2 else 0
                for cls_feature2 in classes.unique()
            ]
            class_dist.append(class_row)
        distances.append(class_dist)

    two_d_mat = [flatten(distances[idx]) for idx in range(len(distances))]
    df_dists = pd.DataFrame(two_d_mat)
    dist_dict = {f'feature_{idx + 1}': pd.DataFrame(mat) for idx, mat in enumerate(distances)}
    return df_dists, dist_dict


def features_reduction(all_features, dists_dict, features_to_reduce_prc):
    import pandas as pd
    from utils.general import calc_k

    features_to_reduce_num = calc_k(all_features, features_to_reduce_prc)
    features_to_reduce_df = pd.DataFrame({'features': [*range(0, len(all_features), 1)], 'count': len(all_features) * [0]})
    for dist, df_dists in dists_dict.items():
        features_to_keep_idx, features_to_reduce_idx = dist_features_reduction(df_dists, features_to_reduce_prc)
        for feature in features_to_reduce_idx:
            features_to_reduce_df.at[feature, 'count'] += 1

    features_to_reduce_df.sort_values(by='count', ascending=False, inplace=True)
    final_features_to_reduce_idx = set(features_to_reduce_df.iloc[:features_to_reduce_num]['features'].tolist())
    final_features_to_keep_idx = list(set(df_dists.index).difference(final_features_to_reduce_idx))
    final_dists_dict = {key: value.iloc[final_features_to_keep_idx] for key, value in dists_dict.items()}
    return final_dists_dict, final_features_to_keep_idx


def dist_features_reduction(df_dists, features_to_reduce_prc):
    import numpy as np

    df_feature_avg = df_dists.mean(axis=1)
    num_features_to_reduce = int(len(df_dists) * features_to_reduce_prc)
    features_to_keep_idx = df_feature_avg.iloc[np.argsort(df_feature_avg)][num_features_to_reduce:].index.sort_values()
    features_to_reduce_idx = list(set(df_feature_avg.index).difference(features_to_keep_idx))
    return features_to_keep_idx, features_to_reduce_idx
