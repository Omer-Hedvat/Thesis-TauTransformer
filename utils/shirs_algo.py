import numpy as np
import pandas as pd
import logging
from sklearn.cluster import KMeans

from utils.distances import wasserstein_dist, bhattacharyya_dist, hellinger_dist, jm_dist
from utils.general import flatten, calc_k

logger = logging.getLogger(__name__)


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
    features = X_tr.columns
    df = X_tr
    classes.reset_index(drop=True, inplace=True)
    df[label_column] = classes
    distances = []
    for feature in features:
        class_dist = []
        for cls_feature1 in classes.unique():
            class_row = [
                execute_distance_func(df, dist_func_name, feature, cls_feature1, cls_feature2)
                if cls_feature1 != cls_feature2 else 0
                for cls_feature2 in classes.unique()
            ]
            class_dist.append(class_row)
        distances.append(class_dist)

    two_d_mat = [flatten(distances[idx]) for idx in range(len(distances))]
    df_dists = pd.DataFrame(two_d_mat)
    dist_dict = {f'feature_{idx + 1}': pd.DataFrame(mat) for idx, mat in enumerate(distances)}
    return df_dists, dist_dict


def features_reduction(all_features, dists_dict, features_to_reduce_prc, verbose=False):
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

    if verbose:
        logger.info(f'features_reduction() -  By a majority of votes, a {features_to_reduce_prc}%, {features_to_reduce_num} features reduction of features has been to:\n{all_features[list(final_features_to_reduce_idx)]}')
    return final_dists_dict, final_features_to_keep_idx


def dist_features_reduction(df_dists, features_to_reduce_prc):
    df_feature_avg = df_dists.mean(axis=1)
    num_features_to_reduce = int(len(df_dists) * features_to_reduce_prc)
    features_to_keep_idx = df_feature_avg.iloc[np.argsort(df_feature_avg)][num_features_to_reduce:].index.sort_values()
    features_to_reduce_idx = list(set(df_feature_avg.index).difference(features_to_keep_idx))
    return features_to_keep_idx, features_to_reduce_idx


def return_best_features_by_kmeans(coordinates, k):
    features_rank = np.argsort(coordinates[0])
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit(coordinates.T).labels_
    best_features = []
    selected_cetroids = []
    for idx in features_rank:
        if labels[idx] not in selected_cetroids:
            selected_cetroids.append(labels[idx])
            best_features.append(idx)
    return best_features, labels, features_rank