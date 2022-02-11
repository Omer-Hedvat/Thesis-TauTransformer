import os
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import logging
from utils.distances import norm_by_dist_type, calculate_distance, wasserstein_dist, bhattacharyya_dist, hellinger_dist, jm_dist
from utils.general import flatten, setup_logger, calc_mean_std
from utils.machine_learning import min_max_scaler
from datetime import datetime
from math import exp, sqrt, log
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics

from dictances import jensen_shannon
from pydiffmap import diffusion_map as dm
from pydiffmap.visualization import embedding_plot, data_plot

from ref.diffusion_maps import diffusion_mapping
from ref.Shir import utils as shir_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold, KFold, StratifiedKFold

logger = logging.getLogger(__name__)


def return_farest_features_from_center(coordinates, k):
    dist = []
    for c in coordinates.T:
        dist.append(sqrt((c[0] ** 2) + (c[1] ** 2)))
    ranking_idx = np.argsort(dist)
    return ranking_idx[-k:]


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
    assert function_name in ['wasserstein_dist', 'bhattacharyya_dist', 'jm_dist', 'hellinger_dist']
    return {
        'wasserstein_dist': lambda: wasserstein_dist(df, feature, label1, label2),
        'bhattacharyya_dist': lambda: bhattacharyya_dist(df, feature, label1, label2),
        'hellinger_dist': lambda: hellinger_dist(df, feature, label1, label2),
        'jm_dist': lambda: jm_dist(df, feature, label1, label2)
    }[function_name]()


def calc_dist(dist_func_name, X_tr, classes):
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
    df['label'] = classes
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


def export_heatmaps(df, features, dist_type1, dist_type2, to_norm=False):
    assert dist_type1 in (
        'wasserstein_dist', 'bhattacharyya_dist', 'jensen_shannon_dist', 'hellinger_dist', 'jm_dist')
    assert dist_type2 in (
        'wasserstein_dist', 'bhattacharyya_dist', 'jensen_shannon_dist', 'hellinger_dist', 'jm_dist')
    _, dist_dict1 = calc_dist(dist_type1, df, 'label')
    _, dist_dict2 = calc_dist(dist_type2, df, 'label')

    cols = [dist_type1, dist_type2]
    rows = ['feature {}'.format(row) for row in features]
    fig, axes = plt.subplots(nrows=len(features), ncols=2, figsize=(8, 25))

    for i, feature in zip(range(len(rows)), features):
        feature_mat1 = dist_dict1[feature]
        feature_mat2 = dist_dict2[feature]
        if to_norm:
            feature_dist_mat1 = norm_by_dist_type(feature_mat1)
            feature_dist_mat2 = norm_by_dist_type(feature_mat2)
        else:
            feature_dist_mat1 = feature_mat1
            feature_dist_mat2 = feature_mat2
        sns.heatmap(feature_dist_mat1, annot=True, linewidths=.5, ax=axes[i, 0])
        sns.heatmap(feature_dist_mat2, annot=True, linewidths=.5, ax=axes[i, 1])

    for axc, col in zip(axes[0], cols):
        axc.set_title(col)

    for axr, row in zip(axes[:, 0], rows):
        axr.set_ylabel(row, rotation=90, size='large')

    plt.show()


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


def k_medoids_features(coordinates, k):
    # calc KMediod to get to centers
    coordinates = coordinates.T
    kmedoids = KMedoids(n_clusters=k, random_state=0).fit(coordinates)
    centers = kmedoids.cluster_centers_

    # search for the features index
    r_features = []
    for i, v in enumerate(coordinates):
        if v in centers:
            r_features.append(i)
    return r_features


def predict(X_train, y_train, X_test=None, y_test=None):
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    clf = RandomForestClassifier(random_state=1)
    multi_target_forest = OneVsRestClassifier(clf, n_jobs=-1)
    train_acc = []

    for train_index, test_index in kf.split(X_train, y_train):
        model = multi_target_forest.fit(X_train.iloc[train_index], y_train.iloc[train_index])
        train_preds = model.predict(X_train.iloc[test_index])

        train_acc.append(metrics.accuracy_score(y_train.iloc[test_index], train_preds))
    if X_test is not None and y_test is not None:
        model = multi_target_forest.fit(X_train, y_train)
        preds = model.predict(X_test)
        logger.info(metrics.classification_report(y_test, preds, digits=3))

    train_avg_score = sum(train_acc) / len(train_acc)
    logger.info(f"Cross validation accuracies = {train_acc}")
    logger.info(f"Cross validation average accuracy = {train_avg_score}\n")
    return train_acc


def predict_np(X_tr, X_tst, y_train, y_test):
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    clf = RandomForestClassifier(random_state=1)
    multi_target_forest = OneVsRestClassifier(clf, n_jobs=-1)
    train_acc = []

    for train_index, test_index in kf.split(X_tr, y_train):
        model = multi_target_forest.fit(X_tr[train_index], y_train[train_index])
        train_preds = model.predict(X_tr[test_index])

        train_acc.append(metrics.accuracy_score(y_train[test_index], train_preds))

    model = multi_target_forest.fit(X_tr, y_train)
    preds = model.predict(X_tst)
    logger.info(metrics.classification_report(y_test, preds, digits=3))
    return train_acc


def calc_k(features, prc):
    return int(len(features) * prc)


def main():
    config = {
        'dataset_name': 'glass',
        'label_column': 'label',
        'features_percentage': 0.5,
        'dist_functions': ['wasserstein_dist', 'hellinger_dist', 'jm_dist'],
        'nrows': 10000,
        'alpha': 1,
        'eps_type': 'maxmin'
    }

    dataset_dir = f"data/{config['dataset_name']}.csv"
    setup_logger("config_files/logger_config.json", os.path.join('results', f"{config['dataset_name']}_log_{datetime.now().strftime('%d-%m-%Y')}.txt"))
    logger.info(f'{dataset_dir=}')
    data = pd.read_csv(dataset_dir, nrows=config['nrows'])

    features = data.columns.drop(config['label_column'])

    k = calc_k(features, config['features_percentage'])

    logger.info(f"DATA STATS:\ndata shape of {data.shape}\nLabel distributes:\n{data.label.value_counts().sort_index()}\n")

    logger.info('*' * 100)
    logger.info(f"{'*' * 37} Using all features prediction {'*' * 37}")
    logger.info('*' * 100)
    X, y = data[features].copy(), data[config['label_column']].copy()
    predict(X, y)

    logger.info(f"Running over {dataset_dir}, using {k} features out of {len(features)}")

    logger.info('*' * 100)
    logger.info(f"{'*' * 40} Using Random {k} features prediction {'*' * 40}")
    logger.info('*' * 100)
    sampled_data = data[features].sample(n=k, axis='columns')
    new_features = sampled_data.columns
    sampled_data[config['label_column']] = data[config['label_column']]
    X, y = sampled_data[new_features].copy(), sampled_data[config['label_column']].copy()
    predict(X, y)

    for dist in config['dist_functions']:
        logger.info('*' * 100)
        logger.info(f"{'*' * 40} {dist} {'*' * 40}")
        logger.info('*' * 100)

        X, y = data[features].copy(), data[config['label_column']].copy()
        # Norm
        X_norm = min_max_scaler(X, features)

        df_dists, dist_dict = calc_dist(dist, X_norm, y)
        vec, egs, coordinates, dataList, epsilon, ranking = (diffusion_mapping(df_dists, config['alpha'], config['eps_type'], 8, 1, dim=2))

        flat_ranking = [item for sublist in ranking for item in sublist]
        ranking_idx = np.argsort(flat_ranking)
        logger.info(f'best features by {dist} are: {ranking_idx}')
        predict(X.iloc[:, ranking_idx[-k:]], y)

        best_features, labels, features_rank = return_best_features_by_kmeans(coordinates, k)
        logger.info(f'Best features by KMeans are: {best_features}')
        predict(X.iloc[:, best_features], y)

        k_features = k_medoids_features(coordinates, k)
        logger.info(f'Best features by KMediods are: {k_features}')
        predict(X.iloc[:, k_features], y)

        best_features = return_farest_features_from_center(coordinates, k)
        logger.info(f'best features by farest coordinate from (0,0) are: {ranking_idx}')
        predict(X.iloc[:, best_features], y)


if __name__ == '__main__':
    main()
