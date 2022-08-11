from datetime import datetime
import itertools
import logging
import os

from TauTransformer import TauTransformer
from utils.diffusion_maps import diffusion_mapping
from utils.distances import calc_dist, features_reduction
from utils.files import create_work_dir, read_from_csv, print_separation_dots, store_results, all_results_colorful
from utils.general import setup_logger, lists_avg, calc_k, arrange_data_features
from utils.machine_learning import min_max_scaler, t_test, kfolds_split
from utils.machine_learning import (
    predict, random_features_predict, fisher_ranks_predict, relieff_predict, chi_square_predict, mrmr_predict, return_best_features_by_kmeans
)
from utils.timer import Timer


logger = logging.getLogger(__name__)


def run_experiments(config, api_params):
    workdir = os.path.join(f'results', config['dataset_name'])
    create_work_dir(workdir, on_exists='ignore')
    setup_logger("config_files/logger_config.json", os.path.join(workdir, f"{config['dataset_name']}_log_{datetime.now().strftime('%d-%m-%Y')}.txt"))
    dataset_dir = f"data/{config['dataset_name']}.csv"

    logger.info(f'{dataset_dir=}')
    data, config['dataset_name'] = read_from_csv(dataset_dir, config)
    all_features = data.columns.drop('label')
    classes = list(data['label'].unique())
    logger.info(f"DATA STATS:\ndata shape of {data.shape}\nLabel distributes:\n{data['label'].value_counts().sort_index()}\n")

    all_features_acc_agg, all_features_f1_agg, timer_all_features = [], [], []
    print_separation_dots('Using all features prediction')
    for kfold_iter in range(1, config['kfolds'] + 1):
        train_set, val_set = kfolds_split(data, kfold_iter, n_splits=config['kfolds'], random_state=0)
        X_tr, y_tr, X_test, y_test = arrange_data_features(train_set, val_set, all_features, return_y=True)

        with Timer() as timer:
            all_features_acc, all_features_f1 = predict(X_tr, y_tr, X_test, y_test)
        all_features_acc_agg.append(all_features_acc)
        all_features_f1_agg.append(all_features_f1)
        timer_all_features.append(timer)
    for feature_percentage, dm_dim in list(itertools.product(config['features_percentage'], config['dm_dim'])):
        api_params['dm_dim'] = dm_dim
        k = calc_k(all_features, feature_percentage)
        if k < 1 or k == len(all_features):
            continue
        logger.info(f"""
        Running over features percentage of {feature_percentage}, which is {k} features out of {data.shape[1] - 1}, with diffusion mapping dimension of {dm_dim}"""
                    )

        # Init results lists
        print_separation_dots(f'Starting baseline heuristics using: random features, Fisher score, ReliefF selection, Chi-square Test selection &  & mRMR for {k} features out of {len(all_features)}')
        random_acc_agg, random_f1_agg, timer_random = [], [], []
        fisher_acc_agg, fisher_f1_agg, timer_fisher = [], [], []
        relief_acc_agg, relief_f1_agg, timer_relief = [], [], []
        chi2_acc_agg, chi2_f1_agg, timer_chi2 = [], [], []
        mrmr_acc_agg, mrmr_f1_agg, timer_mrmr = [], [], []
        jm_kmeans_acc_agg, jm_kmeans_f1_agg, timer_jm = [], [], []
        for kfold_iter in range(1, config['kfolds'] + 1):
            final_kf_iter = kfold_iter == config['kfolds']
            train_set, val_set = kfolds_split(data, kfold_iter, n_splits=config['kfolds'], random_state=config['random_state'])

            # Storing the results we've calculated earlier for all_features
            if final_kf_iter:
                acc_result = round(lists_avg(all_features_acc_agg)*100, 2)
                logger.info(f"all_features accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, dm_dim, 'all_features', all_features_acc_agg, all_features_f1_agg, classes, workdir, timer_all_features)

            # Random Features
            with Timer() as timer:
                random_acc_agg, random_f1_agg = random_features_predict(train_set, val_set, k, all_features, random_acc_agg, random_f1_agg, config['random_state'])
            timer_random.append(timer)
            if final_kf_iter:
                acc_result = round(lists_avg(random_acc_agg) * 100, 2)
                logger.info(f"random_features accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, dm_dim, 'random_features', random_acc_agg, random_f1_agg, classes, workdir, timer_random)

            # Fisher Score Features
            with Timer() as timer:
                fisher_acc_agg, fisher_f1_agg = fisher_ranks_predict(train_set, val_set, k, all_features, fisher_acc_agg, fisher_f1_agg)
            timer_fisher.append(timer)
            if final_kf_iter:
                acc_result = round(lists_avg(fisher_acc_agg) * 100, 2)
                logger.info(f"fisher_score accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, dm_dim, 'fisher', fisher_acc_agg, fisher_f1_agg, classes, workdir, timer_fisher)

            # ReliefF Features
            with Timer() as timer:
                relief_acc_agg, relief_f1_agg = relieff_predict(train_set, val_set, k, all_features, relief_acc_agg, relief_f1_agg)
            timer_relief.append(timer)
            if final_kf_iter:
                acc_result = round(lists_avg(relief_acc_agg) * 100, 2)
                logger.info(f"Relief accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, dm_dim, 'relief', relief_acc_agg, relief_f1_agg, classes, workdir, timer_relief)

            # Chi Square Features
            with Timer() as timer:
                chi2_acc_agg, chi2_f1_agg = chi_square_predict(train_set, val_set, k, all_features, chi2_acc_agg, chi2_f1_agg)
            timer_chi2.append(timer)
            if final_kf_iter:
                acc_result = round(lists_avg(chi2_acc_agg) * 100, 2)
                logger.info(f"chi_square accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, dm_dim, 'chi_square', chi2_acc_agg, chi2_f1_agg, classes, workdir, timer_chi2)

            # mRMR Features
            with Timer() as timer:
                mrmr_acc_agg, mrmr_f1_agg = mrmr_predict(train_set, val_set, k, all_features, mrmr_acc_agg, mrmr_f1_agg)
            timer_mrmr.append(timer)
            if final_kf_iter:
                acc_result = round(lists_avg(mrmr_acc_agg) * 100, 2)
                logger.info(f"mRMR accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, dm_dim, 'mrmr', mrmr_acc_agg, mrmr_f1_agg, classes, workdir, timer_mrmr)

            # Shir's Approach Features
            with Timer() as timer:
                jm_dict = {}
                X_tr_norm = min_max_scaler(train_set, all_features)
                jm_dists, _ = calc_dist('jm', X_tr_norm, y_tr, 'label')
                jm_dict['jm'] = jm_dists
                if feature_percentage + 0.5 < 1:
                    jm_distances_dict, _ = features_reduction(all_features, jm_dict, 0.5)
                else:
                    jm_distances_dict = jm_dict.copy()
                jm_coordinates, jm_ranking = diffusion_mapping(jm_distances_dict['jm'], config['alpha'], config['eps_type'], config['eps_factor'], dim=dm_dim)

                jm_features, _, _ = return_best_features_by_kmeans(jm_coordinates, k)
                X_tr, X_test = arrange_data_features(train_set, val_set, all_features, return_y=False)
                jm_kmeans_acc, jm_kmeans_f1 = predict(X_tr.iloc[:, jm_features], y_tr, X_test.iloc[:, jm_features], y_test)
            jm_kmeans_acc_agg.append(jm_kmeans_acc)
            jm_kmeans_f1_agg.append(jm_kmeans_f1)
            timer_jm.append(timer)
            if final_kf_iter:
                jm_acc_result = round(lists_avg(jm_kmeans_acc_agg) * 100, 2)
                logger.info(f"Shir's algo kmeans accuracy result w/ {int(0.5 * 100)}% huristic: {jm_acc_result}%")
                store_results(config['dataset_name'], feature_percentage, dm_dim, 'shirs_algo', jm_kmeans_acc_agg, jm_kmeans_f1_agg, classes, workdir, timer_jm)

        # TauTransformer Features
        for features_to_reduce_prc in config['features_to_reduce_prc']:
            if feature_percentage + features_to_reduce_prc >= 1:
                continue
            print_separation_dots(f'features to reduce heuristic of {int(features_to_reduce_prc*100)}%')

            kmeans_acc_agg, kmeans_f1_agg, timer_tau_trans = [], [], []
            for kfold_iter in range(1, config['kfolds'] + 1):
                final_kf_iter = kfold_iter == config['kfolds']
                train_set, val_set = kfolds_split(data, kfold_iter, n_splits=config['kfolds'], random_state=0)
                X_train, y_train, X_test, y_test = arrange_data_features(train_set, val_set, all_features)

                with Timer() as timer:
                    tt = TauTransformer(feature_percentage, features_to_reduce_prc, config['dist_functions'], **api_params)
                    X_tr = tt.fit_transform(X_train, y_train)
                    X_tst = tt.transform(X_test)
                    kmeans_acc, kmeans_f1 = predict(X_tr, y_train, X_tst, y_test)

                kmeans_acc_agg.append(kmeans_acc)
                kmeans_f1_agg.append(kmeans_f1)
                timer_tau_trans.append(timer)
                if final_kf_iter:
                    acc_result = round(lists_avg(kmeans_acc_agg) * 100, 2)
                    logger.info(f"kmeans accuracy result w/ {int(features_to_reduce_prc*100)}% huristic: {acc_result}%")
                    store_results(config['dataset_name'], feature_percentage, dm_dim, f'kmeans_{features_to_reduce_prc}', kmeans_acc_agg, kmeans_f1_agg, classes, workdir, timer_tau_trans)

    t_test(config['dataset_name'])


def main():
    config = {
        'kfolds': 5,
        'features_percentage': [0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
        'dist_functions': ['wasserstein', 'jm', 'hellinger'],
        'nrows': 10000,
        'features_to_reduce_prc': [0.0, 0.2, 0.35, 0.5],
        'dm_dim': [2],
        'alpha': 1,
        'eps_type': 'maxmin',
        'eps_factor': 25,
        'verbose': False,
        'random_state': 0,
        'add_features_up_to':100
    }

    api_params = {
        'alpha': config['alpha'],
        'eps_type': config['eps_type'],
        'eps_factor': config['eps_factor'],
        'verbose': config['verbose'],
        'random_state': config['random_state']
    }

    # tuples of datasets names and target column name
    datasets = [
        ('adware_balanced', 'label'), ('ml_multiclass_classification_data', 'target'), ('digits', 'label'), ('isolet', 'label'),
        ('otto_balanced', 'target'), ('gene_data', 'label')
    ]
    datasets = [('adware_balanced', 'label')]
    config['features_percentage'] = [0.02, 0.05, 0.1, 0.2, 0.3]
    config['features_to_reduce_prc'] = [0.0, 0.2, 0.35, 0.5]

    for dataset, label in datasets:
        config['dataset_name'] = dataset
        config['label_column'] = label
        run_experiments(config, api_params)

    all_results_colorful()


if __name__ == '__main__':
    main()
