from datetime import datetime
import itertools
import logging
import numpy as np
import os
import pandas as pd

from sklearn.cluster import KMeans

from TauTransformer import TauTransformer
from utils.diffusion_maps import diffusion_mapping
from utils.distances import wasserstein_dist, bhattacharyya_dist, hellinger_dist, jm_dist
from utils.files import create_work_dir, read_from_csv, print_separation_dots, store_results, all_results_colorful
from utils.general import flatten, setup_logger, lists_avg, calc_k, train_test_split
from utils.machine_learning import min_max_scaler, t_test, kfolds_split
from utils.machine_learning import (
    predict, random_features_predict, fisher_ranks_predict, relieff_predict, chi_square_predict, mrmr_predict
)


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
        # random_acc_agg, random_f1_agg = [], []
        # fisher_acc_agg, fisher_f1_agg = [], []
        # relief_acc_agg, relief_f1_agg = [], []
        # chi_square_acc_agg, chi_square_f1_agg = [], []
        # mrmr_acc_agg, mrmr_f1_agg = [], []
        # jm_kmeans_acc_agg, jm_kmeans_f1_agg = [], []
        # for kfold_iter in range(1, config['kfolds'] + 1):
        #     final_kf_iter = kfold_iter == config['kfolds']
        #     train_set, val_set = kfolds_split(data, kfold_iter, n_splits=config['kfolds'], random_state=config['random_state'])
        #
        #     # Storing the results we've calculated earlier for all_features
        #     if final_kf_iter:
        #         acc_result = round(lists_avg(all_features_acc_agg)*100, 2)
        #         logger.info(f"all_features accuracy result: {acc_result}%")
        #         store_results(config['dataset_name'], feature_percentage, dm_dim, 'all_features', all_features_acc_agg, all_features_f1_agg, classes, workdir)
        #
        #     # Random Features
        #     random_acc_agg, random_f1_agg = random_features_predict(train_set, val_set, k, all_features, random_acc_agg, random_f1_agg, config['random_state'])
        #     if final_kf_iter:
        #         acc_result = round(lists_avg(random_acc_agg) * 100, 2)
        #         logger.info(f"random_features accuracy result: {acc_result}%")
        #         store_results(config['dataset_name'], feature_percentage, dm_dim, 'random_features', random_acc_agg, random_f1_agg, classes, workdir)
        #
        #     # Fisher Score Features
        #     fisher_acc_agg, fisher_f1_agg = fisher_ranks_predict(train_set, val_set, k, all_features, fisher_acc_agg, fisher_f1_agg)
        #     if final_kf_iter:
        #         acc_result = round(lists_avg(fisher_acc_agg) * 100, 2)
        #         logger.info(f"fisher_score accuracy result: {acc_result}%")
        #         store_results(config['dataset_name'], feature_percentage, dm_dim, 'fisher', fisher_acc_agg, fisher_f1_agg, classes, workdir)
        #
        #     # ReliefF Features
        #     relief_acc_agg, relief_f1_agg = relieff_predict(train_set, val_set, k, all_features, relief_acc_agg, relief_f1_agg)
        #     if final_kf_iter:
        #         acc_result = round(lists_avg(relief_acc_agg) * 100, 2)
        #         logger.info(f"Relief accuracy result: {acc_result}%")
        #         store_results(config['dataset_name'], feature_percentage, dm_dim, 'relief', relief_acc_agg, relief_f1_agg, classes, workdir)
        #
        #     # Chi Suare Features
        #     chi_square_acc_agg, chi_square_f1_agg = chi_square_predict(train_set, val_set, k, all_features, chi_square_acc_agg, chi_square_f1_agg)
        #     if final_kf_iter:
        #         acc_result = round(lists_avg(chi_square_acc_agg) * 100, 2)
        #         logger.info(f"chi_square accuracy result: {acc_result}%")
        #         store_results(config['dataset_name'], feature_percentage, dm_dim, 'chi_square', chi_square_acc_agg, chi_square_f1_agg, classes, workdir)
        #
        #     # mRMR Features
        #     mrmr_acc_agg, mrmr_f1_agg = mrmr_predict(train_set, val_set, k, all_features, mrmr_acc_agg, mrmr_f1_agg)
        #     if final_kf_iter:
        #         acc_result = round(lists_avg(mrmr_acc_agg) * 100, 2)
        #         logger.info(f"mRMR accuracy result: {acc_result}%")
        #         store_results(config['dataset_name'], feature_percentage, dm_dim, 'mrmr', mrmr_acc_agg,
        #                       mrmr_f1_agg, classes, workdir)

            # # Shir's Approach Features
            # jm_dict = {}
            # X_tr_norm = min_max_scaler(train_set, all_features)
            # jm_dists, _ = calc_dist('jm', X_tr_norm, y_tr, 'label')
            # jm_dict['jm'] = jm_dists
            # if feature_percentage + 0.5 < 1:
            #     jm_distances_dict, _ = features_reduction(all_features, jm_dict, 0.5, config['verbose'])
            # else:
            #     jm_distances_dict = jm_dict.copy()
            # jm_coordinates, jm_ranking = diffusion_mapping(jm_distances_dict['jm'], config['alpha'], config['eps_type'], config['eps_factor'], dim=dm_dim)
            #
            # jm_features, _, _ = return_best_features_by_kmeans(jm_coordinates, k)
            # X_tr, X_test = train_test_split(train_set, val_set, all_features, return_y=False)
            # jm_kmeans_acc, jm_kmeans_f1 = predict(X_tr.iloc[:, jm_features], y_tr, X_test.iloc[:, jm_features], y_test)
            # jm_kmeans_acc_agg.append(jm_kmeans_acc)
            # jm_kmeans_f1_agg.append(jm_kmeans_f1)
            # if final_kf_iter:
            #     jm_acc_result = round(lists_avg(jm_kmeans_acc_agg) * 100, 2)
            #     logger.info(f"Shir's algo kmeans accuracy result w/ {int(0.5 * 100)}% huristic: {jm_acc_result}%")
            #     store_results(config['dataset_name'], feature_percentage, dm_dim, 'shirs_algo', jm_kmeans_acc_agg, jm_kmeans_f1_agg, classes, workdir)

        # New Approach Features
        for features_to_reduce_prc in config['features_to_reduce_prc']:
            if feature_percentage + features_to_reduce_prc >= 1:
                continue
            print_separation_dots(f'features to reduce heuristic of {int(features_to_reduce_prc*100)}%')

            kmeans_acc_agg, kmeans_f1_agg = [], []
            for kfold_iter in range(1, config['kfolds'] + 1):
                final_kf_iter = kfold_iter == config['kfolds']
                train_set, val_set = kfolds_split(data, kfold_iter, n_splits=config['kfolds'], random_state=0)
                X_tr, y_tr, X_test, y_test = train_test_split(train_set, val_set, all_features)

                tau_trans = TauTransformer(
                    X_tr, y_tr, feature_percentage, features_to_reduce_prc, config['dist_functions'], dm_dim, config['alpha'], config['eps_type'],
                    config['eps_factor'], config['random_state'], config['verbose'])
                best_features, best_features_idx = tau_trans.transform()

                kmeans_acc, kmeans_f1 = predict(X_tr.iloc[:, best_features_idx], y_tr, X_test.iloc[:, best_features_idx], y_test)
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
        'nrows': 1000,
        'features_to_reduce_prc': [0.0, 0.2, 0.35, 0.5],
        'dm_dim': [2],
        'alpha': 1,
        'eps_type': 'maxmin',
        'eps_factor': 25,
        'verbose': False,
        'random_state': 0
    }

    # tuples of datasets names and target column name
    datasets = [
        ('adware_balanced', 'label'), ('ml_multiclass_classification_data', 'target'), ('digits', 'label'), ('isolet', 'label'),
        ('otto_balanced', 'target'), ('gene_data', 'label')
    ]
    # datasets = [('otto_balanced', 'target')]
    # config['features_percentage'] = [0.02, 0.1, 0.3]
    # config['features_to_reduce_prc']: [0.0, 0.2]

    for dataset, label in datasets:
        config['dataset_name'] = dataset
        config['label_column'] = label
        run_experiments(config)

    all_results_colorful()


if __name__ == '__main__':
    main()
