from datetime import datetime
import itertools
import logging
from math import sqrt
import numpy as np
import os
import pandas as pd
import random

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter

from mrmr import mrmr_classif
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from skfeature.function.similarity_based import fisher_score
from ReliefF import ReliefF
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy import stats

from utils.diffusion_maps import diffusion_mapping
from utils.distances import wasserstein_dist, bhattacharyya_dist, hellinger_dist, jm_dist
from utils.files import create_work_dir, read_from_csv, print_separation_dots
from utils.general import flatten, setup_logger, lists_avg, calc_k, train_test_split
from utils.machine_learning import min_max_scaler, kfolds_split

logger = logging.getLogger(__name__)


def predict(X_train, y_train, X_val, y_val):
    clf = RandomForestClassifier(random_state=1)
    multi_target_forest = OneVsRestClassifier(clf, n_jobs=-1)
    model = multi_target_forest.fit(X_train, y_train)
    validation_preds = model.predict(X_val)

    train_acc = metrics.accuracy_score(y_val, validation_preds)
    f1_scores_list = metrics.f1_score(y_val, validation_preds, average=None)
    return train_acc, f1_scores_list


def calc_f1_score(f1_lists):
    return list(np.array(f1_lists).mean(axis=0))


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


def store_results(dataset, features_prc, dm_dim, metric, acc, f1, classes, workdir):
    # General Results File
    acc_results_df = pd.read_csv('results/all_datasets_results.csv')
    ds_results_mask = (
            (acc_results_df.dataset == dataset) & (acc_results_df.features_prc == features_prc) &
            (acc_results_df.dm_dim == dm_dim)
    )
    if ds_results_mask.any():
        acc_results_df.loc[ds_results_mask, metric] = lists_avg(acc)
    else:
        today_date = datetime.now().strftime('%d-%m-%Y')
        new_df = pd.DataFrame(columns=acc_results_df.columns)
        new_df.loc[len(new_df), ['date', 'dataset', 'features_prc', 'dm_dim', metric]] = [today_date, dataset, features_prc, dm_dim, lists_avg(acc)]
        acc_results_df = pd.concat([acc_results_df, new_df]).sort_values(by=['dataset', 'features_prc', 'dm_dim'])
    acc_results_df.to_csv('results/all_datasets_results.csv', index=False)

    # Dataset's F1 Results File
    columns = ['features_prc', 'dm_dim', *[f'{metric}_{class_name}' for class_name in classes]]
    class_avg_f1 = calc_f1_score(f1)
    values = [features_prc, dm_dim, *class_avg_f1]
    data_dict = dict(zip(columns, values))
    f1_file = os.path.join(workdir, f'f1_scores.csv')
    new_data_df = pd.DataFrame([data_dict])
    if not os.path.exists(f1_file):
        new_data_df.to_csv(f1_file, index=False)
    else:
        f1_results_df = pd.read_csv(f1_file)
        all_ds_results_mask = ((f1_results_df.features_prc == features_prc) & (f1_results_df.dm_dim == dm_dim))
        if all_ds_results_mask.any():
            f1_results_df.loc[all_ds_results_mask, columns] = values
        else:
            f1_results_df = pd.concat([f1_results_df, new_data_df]).sort_values(by=['features_prc', 'dm_dim'])
        f1_results_df.to_csv(f1_file, index=False)


def t_test(dataset_name):
    """
    :param dataset_name: the name of the dataset we are using
    :return: add all t_test p-valus for the dataset.
    The T test calculation is done for each of our methods versus the rest of our conventional
    methods we compare: 'random_features', 'fisher', 'relief', 'chi_square'
    """
    data = pd.read_csv('results/all_datasets_results.csv')
    data = data[data['dataset'] == dataset_name]
    A_type = ['random_features', 'fisher', 'relief', 'chi_square']
    B_type = ['kmeans_0.0', 'kmeans_0.2', 'kmeans_0.35', 'kmeans_0.5']
    df = pd.DataFrame(data={'dataset': [dataset_name]})
    for a in A_type:
        for b in B_type:
            # t test is A>B
            df[f'{a}_vs_{b}'] = stats.ttest_rel(data[a], data[b], alternative='less')[1]
    old_df = pd.read_csv('results/t_test_results.csv')
    df = pd.concat([df, old_df], ignore_index=True)
    df.to_csv('results/t_test_results.csv', index=False)


def all_results_colorful():
    data = pd.read_csv("results/all_datasets_results.csv")
    dat = data['dataset'] + " " + data['features_prc'].apply(str) + " " + data['dm_dim'].apply(str)
    data['raw'] = dat
    data = data.set_index('raw')
    data = data.drop(columns=['date', 'dataset', 'features_prc', 'dm_dim'])
    data.style.background_gradient(cmap='RdYlGn', axis=1).to_excel("results/all_resualts_colors.xlsx")


def run_experiments(config):
    workdir = os.path.join(f'results', config['dataset_name'])
    create_work_dir(workdir, on_exists='ignore')
    setup_logger("config_files/logger_config.json", os.path.join(workdir, f"{config['dataset_name']}_log_{datetime.now().strftime('%d-%m-%Y')}.txt"))
    dataset_dir = f"data/{config['dataset_name']}.csv"

    logger.info(f'{dataset_dir=}')
    data = read_from_csv(dataset_dir, config)
    all_features = data.columns.drop('label')
    classes = list(data['label'].unique())
    logger.info(f"DATA STATS:\ndata shape of {data.shape}\nLabel distributes:\n{data['label'].value_counts().sort_index()}\n")

    all_features_acc_agg, all_features_f1_agg = [], []
    print_separation_dots('Using all features prediction')
    for kfold_iter in range(1, config['kfolds'] + 1):
        train_set, val_set = kfolds_split(data, kfold_iter, n_splits=config['kfolds'], random_state=0)
        X_tr, y_tr, X_test, y_test = train_test_split(train_set, val_set, all_features, return_y=True)

        all_features_acc, all_features_f1 = predict(X_tr, y_tr, X_test, y_test)
        all_features_acc_agg.append(all_features_acc)
        all_features_f1_agg.append(all_features_f1)

    for feature_percentage, dm_dim in list(itertools.product(config['features_percentage'], config['dm_dim'])):
        k = calc_k(all_features, feature_percentage)
        if k < 1 or k == len(all_features):
            continue
        logger.info(f"""
        Running over features percentage of {feature_percentage}, which is {k} features out of {data.shape[1] - 1}, with diffusion mapping dimension of {dm_dim}"""
                    )

        # Init results lists
        print_separation_dots(f'Starting baseline heuristics using: random features, Fisher score, ReliefF selection, Chi-square Test selection &  & mRMR for {k} features out of {len(all_features)}')
        random_acc_agg, random_f1_agg = [], []
        fisher_acc_agg, fisher_f1_agg = [], []
        relief_acc_agg, relief_f1_agg = [], []
        chi_square_acc_agg, chi_square_f1_agg = [], []
        mrmr_acc_agg, mrmr_f1_agg = [], []
        for kfold_iter in range(1, config['kfolds'] + 1):
            final_kf_iter = kfold_iter == config['kfolds']
            train_set, val_set = kfolds_split(data, kfold_iter, n_splits=config['kfolds'], random_state=0)

            # Storing the results we've calculated earlier
            if final_kf_iter:
                acc_result = round(lists_avg(all_features_acc_agg)*100, 2)
                logger.info(f"all_features accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, dm_dim, 'all_features', all_features_acc_agg, all_features_f1_agg, classes, workdir)

            random_features = random.sample(list(all_features), k)
            X_tr, X_test = train_test_split(train_set, val_set, random_features, return_y=False)

            random_acc, random_f1 = predict(X_tr, y_tr, X_test, y_test)
            random_acc_agg.append(random_acc)
            random_f1_agg.append(random_f1)
            if final_kf_iter:
                acc_result = round(lists_avg(random_acc_agg) * 100, 2)
                logger.info(f"random_features accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, dm_dim, 'random_features', random_acc_agg, random_f1_agg, classes, workdir)

            fisher_ranks = fisher_score.fisher_score(train_set[all_features].to_numpy(), train_set['label'].to_numpy())
            fisher_features_idx = np.argsort(fisher_ranks, 0)[::-1][:k]
            fisher_features = all_features[fisher_features_idx]
            X_tr, X_test = train_test_split(train_set, val_set, fisher_features, return_y=False)

            fisher_acc, fisher_f1 = predict(X_tr, y_tr, X_test, y_test)
            fisher_acc_agg.append(fisher_acc)
            fisher_f1_agg.append(fisher_f1)
            if final_kf_iter:
                acc_result = round(lists_avg(fisher_acc_agg) * 100, 2)
                logger.info(f"fisher_score accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, dm_dim, 'fisher', fisher_acc_agg, fisher_f1_agg, classes, workdir)

            fs = ReliefF(n_neighbors=1, n_features_to_keep=k)
            X_tr = fs.fit_transform(train_set[all_features].to_numpy(), train_set['label'].to_numpy())
            X_test = fs.transform(val_set[all_features].to_numpy())

            relief_acc, relief_f1 = predict(X_tr, y_tr, X_test, y_test)
            relief_acc_agg.append(relief_acc)
            relief_f1_agg.append(relief_f1)
            if final_kf_iter:
                acc_result = round(lists_avg(relief_acc_agg) * 100, 2)
                logger.info(f"Relief accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, dm_dim, 'relief', relief_acc_agg, relief_f1_agg, classes, workdir)

            chi_features = SelectKBest(chi2, k=k)
            X_tr_norm, X_test_norm = min_max_scaler(train_set, all_features, val_set, return_as_df=False)
            X_tr = chi_features.fit_transform(X_tr_norm, y_tr)
            X_test = chi_features.transform(X_test_norm)

            chi2_acc, chi2_f1 = predict(X_tr, y_tr, X_test, y_test)
            chi_square_acc_agg.append(chi2_acc)
            chi_square_f1_agg.append(chi2_f1)
            if final_kf_iter:
                acc_result = round(lists_avg(chi_square_acc_agg) * 100, 2)
                logger.info(f"chi_square accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, dm_dim, 'chi_square', chi_square_acc_agg, chi_square_f1_agg, classes, workdir)

            mrmr_features = mrmr_classif(X=train_set[all_features], y=train_set['label'], K=k)
            X_tr, X_test = train_test_split(train_set, val_set, mrmr_features, return_y=False)
            mrmr_acc, mrmr_f1 = predict(X_tr, y_tr, X_test, y_test)
            mrmr_acc_agg.append(mrmr_acc)
            mrmr_f1_agg.append(mrmr_f1)

            if final_kf_iter:
                acc_result = round(lists_avg(mrmr_acc_agg) * 100, 2)
                logger.info(f"mRMR accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, dm_dim, 'mRMR', mrmr_acc_agg,
                              mrmr_f1_agg, classes, workdir)

        for features_to_reduce_prc in config['features_to_reduce_prc']:
            if feature_percentage + features_to_reduce_prc >= 1:
                continue
            print_separation_dots(f'features to reduce heuristic of {int(features_to_reduce_prc*100)}%')

            kmeans_acc_agg, kmeans_f1_agg = [], []
            for kfold_iter in range(1, config['kfolds'] + 1):
                final_kf_iter = kfold_iter == config['kfolds']
                train_set, val_set = kfolds_split(data, kfold_iter, n_splits=config['kfolds'], random_state=0)
                X_tr, y_tr, X_test, y_test = train_test_split(train_set, val_set, all_features)

                dm_dict = {}
                dists_dict = dict()
                if config['verbose']:
                    logger.info(f"Calculating distances for {', '.join(config['dist_functions'])}")

                for dist in config['dist_functions']:
                    X_tr_norm = min_max_scaler(X_tr, all_features)
                    df_dists, _ = calc_dist(dist, X_tr_norm, y_tr, 'label')
                    dists_dict[dist] = df_dists

                if config['verbose']:
                    logger.info(f"Reducing {int(features_to_reduce_prc*100)}% features using 'features_reduction()' heuristic")
                if features_to_reduce_prc > 0:
                    distances_dict, features_to_keep_idx = features_reduction(all_features, dists_dict, features_to_reduce_prc, config['verbose'])
                    features = all_features[features_to_keep_idx]
                else:
                    distances_dict = dists_dict.copy()
                    features = all_features

                if config['verbose']:
                    logger.info(f"Calculating diffusion maps over the distance matrix")
                for dist in config['dist_functions']:
                    coordinates, ranking = diffusion_mapping(distances_dict[dist], config['alpha'], config['eps_type'], config['eps_factor'], dim=dm_dim)
                    dm_dict[dist] = {'coordinates': coordinates, 'ranking': ranking}

                if config['verbose']:
                    logger.info(
                        f"""Ranking the {int((1-features_to_reduce_prc)*100)}% remain features using a combined coordinate matrix ('agg_corrdinates'),  
                        inserting 'agg_corrdinates' into a 2nd diffusion map and storing the 2nd diffusion map results into 'final_coordinates'"""
                    )
                agg_coordinates = np.concatenate([val['coordinates'] for val in dm_dict.values()]).T
                final_coordinates, final_ranking = diffusion_mapping(agg_coordinates, config['alpha'], config['eps_type'], config['eps_factor'], dim=dm_dim)
                best_features, labels, features_rank = return_best_features_by_kmeans(final_coordinates, k)

                if config['verbose']:
                    logger.info(f'Best features by KMeans are: {features[best_features]}')
                    logger.info(f"Using KMeans algorithm in order to rank the features who are in final_coordinates")

                kmeans_acc, kmeans_f1 = predict(X_tr.iloc[:, best_features], y_tr, X_test.iloc[:, best_features], y_test)
                kmeans_acc_agg.append(kmeans_acc)
                kmeans_f1_agg.append(kmeans_f1)
                if final_kf_iter:
                    acc_result = round(lists_avg(kmeans_acc_agg) * 100, 2)
                    logger.info(f"kmeans accuracy result w/ {int(features_to_reduce_prc*100)}% huristic: {acc_result}%")
                    store_results(config['dataset_name'], feature_percentage, dm_dim, f'kmeans_{features_to_reduce_prc}', kmeans_acc_agg, kmeans_f1_agg, classes, workdir)

    t_test(config['dataset_name'])


def main():
    config = {
        'kfolds': 5,
        'features_percentage': [0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
        'dist_functions': ['wasserstein', 'jm', 'hellinger'],
        'nrows': 500,
        'features_to_reduce_prc': [0.0, 0.2, 0.35, 0.5],
        'dm_dim': [2],
        'alpha': 1,
        'eps_type': 'maxmin',
        'eps_factor': 25,
        'verbose': False
    }

    # tuples of datasets names and target column name
    datasets = [
        ('adware_balanced', 'label'), ('ml_multiclass_classification_data', 'target'), ('digits', 'label'), ('isolet', 'label'),
        ('otto_balanced', 'target'), ('gene_data', 'label')
    ]
    datasets = [('adware_balanced', 'label')]
    config['features_percentage'] = [0.1]
    config['features_to_reduce_prc']: [0.0, 0.2]

    for dataset, label in datasets:
        config['dataset_name'] = dataset
        config['label_column'] = label
        run_experiments(config)

    all_results_colorful()


if __name__ == '__main__':
    main()
