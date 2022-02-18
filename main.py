from datetime import datetime
import logging
from math import sqrt
import numpy as np
import os
import pandas as pd

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn_extra.cluster import KMedoids

from utils.diffusion_maps import diffusion_mapping
from utils.distances import wasserstein_dist, bhattacharyya_dist, hellinger_dist, jm_dist
from utils.files import create_work_dir, read_from_csv, print_separation_dots
from utils.general import flatten, setup_logger
from utils.machine_learning import min_max_scaler

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
    assert function_name in ['wasserstein', 'bhattacharyya', 'jm', 'hellinger']
    return {
        'wasserstein': lambda: wasserstein_dist(df, feature, label1, label2),
        'bhattacharyya': lambda: bhattacharyya_dist(df, feature, label1, label2),
        'hellinger': lambda: hellinger_dist(df, feature, label1, label2),
        'jm': lambda: jm_dist(df, feature, label1, label2)
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


def store_results(dataset, features_prc, metric, acc, f1, classes, workdir):
    # General Results File
    acc_results_df = pd.read_csv('results/all_datasets_results.csv')
    if ((acc_results_df.dataset == dataset) & (acc_results_df.features_prc == features_prc)).any():
        acc_results_df.loc[(acc_results_df.dataset == dataset) & (acc_results_df.features_prc == features_prc), metric] = sum(acc) / len(acc)
    else:
        today_date = datetime.now().strftime('%d-%m-%Y')
        new_df = pd.DataFrame(columns=acc_results_df.columns)
        new_df.loc[len(new_df), ['date', 'dataset', 'features_prc', metric]] = [today_date, dataset, features_prc, (sum(acc) / len(acc))]
        acc_results_df = pd.concat([acc_results_df, new_df]).sort_values(by=['dataset', 'features_prc'])
    acc_results_df.to_csv('results/all_datasets_results.csv', index=False)

    # Dataset's F1 Results File
    columns = ['features_prc', *[f'{metric}_{class_name}' for class_name in classes]]
    values = [features_prc, *f1]
    f1_file = os.path.join(workdir, f'f1_scores.csv')
    if not os.path.exists(f1_file):
        data_dict = dict(zip(columns, values))
        new_data_df = pd.DataFrame([data_dict])
        new_data_df.to_csv(f1_file, index=False)
    else:
        f1_results_df = pd.read_csv(f1_file)
        f1_results_df.loc[f1_results_df.features_prc == features_prc, columns] = values
        f1_results_df.to_csv(f1_file, index=False)


def predict(X_train, y_train, X_test=None, y_test=None):
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    clf = RandomForestClassifier(random_state=1)
    multi_target_forest = OneVsRestClassifier(clf, n_jobs=-1)
    train_acc = []
    f1_scores_list = []

    for train_index, test_index in kf.split(X_train, y_train):
        model = multi_target_forest.fit(X_train.iloc[train_index], y_train.iloc[train_index])
        train_preds = model.predict(X_train.iloc[test_index])

        train_acc.append(metrics.accuracy_score(y_train.iloc[test_index], train_preds))
        f1_scores_list.append(list(metrics.f1_score(y_train.iloc[test_index], train_preds, average=None)))
    if X_test is not None and y_test is not None:
        model = multi_target_forest.fit(X_train, y_train)
        preds = model.predict(X_test)
        logger.info(metrics.classification_report(y_test, preds, digits=3))

    train_avg_score = sum(train_acc) / len(train_acc)
    logger.info(f"Cross validation accuracies = {train_acc}")
    logger.info(f"Cross validation average accuracy = {train_avg_score}\n")
    return train_acc, f1_scores_list


def calc_f1_score(f1_lists):
    return list(np.array(f1_lists).mean(axis=0))


def calc_k(features, prc):
    return int(len(features) * prc)


def main():
    config = {
        'dataset_name': 'glass',
        'label_column': 'label',
        'features_percentage': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'dist_functions': ['wasserstein', 'hellinger', 'jm'],
        'nrows': 10000,
        'alpha': 1,
        'eps_type': 'maxmin',
        'eps_factor': 25
    }

    workdir = os.path.join(f'results', config['dataset_name'])
    create_work_dir(workdir, on_exists='ignore')
    setup_logger("config_files/logger_config.json", os.path.join(workdir, f"{config['dataset_name']}_log_{datetime.now().strftime('%d-%m-%Y')}.txt"))
    dataset_dir = f"data/{config['dataset_name']}.csv"

    logger.info(f'{dataset_dir=}')
    data = read_from_csv(dataset_dir, config['nrows'])
    features = data.columns.drop(config['label_column'])
    classes = list(data[config['label_column']].unique())

    for feature_percentage in config['features_percentage']:
        k = calc_k(features, feature_percentage)
        if k < 1 or k == len(features):
            continue
        logger.info(f"DATA STATS:\ndata shape of {data.shape}\nLabel distributes:\n{data.label.value_counts().sort_index()}\n")

        print_separation_dots('Using all features prediction')
        X, y = data[features].copy(), data[config['label_column']].copy()
        all_features_acc, all_features_f1 = predict(X, y)
        all_features_f1_agg = calc_f1_score(all_features_f1)
        store_results(config['dataset_name'], feature_percentage, 'all_features', all_features_acc, all_features_f1_agg, classes, workdir)

        logger.info(f"Running over {dataset_dir}, using {k} features out of {len(features)}")
        print_separation_dots(f'Using Random {k} features prediction')
        sampled_data = data[features].sample(n=k, axis='columns')
        new_features = sampled_data.columns
        sampled_data[config['label_column']] = data[config['label_column']]
        X, y = sampled_data[new_features].copy(), sampled_data[config['label_column']].copy()
        random_features_acc, random_features_f1 = predict(X, y)
        random_features_f1_agg = calc_f1_score(random_features_f1)
        store_results(config['dataset_name'], feature_percentage, 'random_features', random_features_acc, random_features_f1_agg, classes, workdir)

        for dist in config['dist_functions']:
            print_separation_dots(f'Using Random {dist} features prediction')

            X, y = data[features].copy(), data[config['label_column']].copy()
            X_norm = min_max_scaler(X, features)

            df_dists, dist_dict = calc_dist(dist, X_norm, y)
            coordinates, ranking = (diffusion_mapping(df_dists, config['alpha'], config['eps_type'], config['eps_factor'], dim=2))

            flat_ranking = [item for sublist in ranking for item in sublist]
            ranking_idx = np.argsort(flat_ranking)
            logger.info(f'best features by {dist} are: {ranking_idx}')
            rank_acc, rank_f1 = predict(X.iloc[:, ranking_idx[-k:]], y)
            rank_f1_agg = calc_f1_score(rank_f1)
            store_results(config['dataset_name'], feature_percentage, f'{dist}_rank', rank_acc, rank_f1_agg, classes, workdir)

            best_features, labels, features_rank = return_best_features_by_kmeans(coordinates, k)
            logger.info(f'Best features by KMeans are: {best_features}')
            kmeans_acc, kmeans_f1 = predict(X.iloc[:, best_features], y)
            kmeans_f1_agg = calc_f1_score(kmeans_f1)
            store_results(config['dataset_name'], feature_percentage, f'{dist}_kmeans', kmeans_acc, kmeans_f1_agg, classes, workdir)

            k_features = k_medoids_features(coordinates, k)
            logger.info(f'Best features by KMediods are: {k_features}')
            kmediods_acc, kmediods_f1 = predict(X.iloc[:, k_features], y)
            kmediods_f1_agg = calc_f1_score(kmediods_f1)
            store_results(config['dataset_name'], feature_percentage, f'{dist}_kmediods', kmediods_acc, kmediods_f1_agg, classes, workdir)

            best_features = return_farest_features_from_center(coordinates, k)
            logger.info(f'best features by farest coordinate from (0,0) are: {ranking_idx}')
            distance_from_0_acc, distance_from_0_f1 = predict(X.iloc[:, best_features], y)
            distance_from_0_f1_agg = calc_f1_score(distance_from_0_f1)
            store_results(config['dataset_name'], feature_percentage, f'{dist}_distance_from_0', distance_from_0_acc, distance_from_0_f1_agg, classes, workdir)


if __name__ == '__main__':
    main()
