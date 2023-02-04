from datetime import datetime
import itertools
import logging
import os

from tausformer_main import main as tausformer_main
from utils.files import create_work_dir, read_from_csv, print_separation_dots, store_results, all_results_colorful
from utils.general import setup_logger, lists_avg, percentage_calculator, arrange_data_features
from utils.machine_learning import t_test, kfolds_split
from utils.machine_learning import (
    predict, random_features_predict, fisher_ranks_predict, relieff_predict, chi_square_predict, mrmr_predict)
from utils.timer import Timer


logger = logging.getLogger(__name__)


def run_experiments(config, dm_params, dataset):
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
    for feature_percentage in config['features_percentage']:
        k = percentage_calculator(feature_percentage, array=all_features)
        if k < 1 or k == len(all_features):
            continue
        logger.info(f"""
        Running over features percentage of {feature_percentage}, which is {k} features out of {data.shape[1] - 1}"""
                    )

        # Init results lists
        print_separation_dots(f'Starting baseline heuristics using: random features, Fisher score, ReliefF selection, Chi-square Test selection &  & mRMR for {k} features out of {len(all_features)}')
        random_acc_agg, random_f1_agg, timer_random = [], [], []
        fisher_acc_agg, fisher_f1_agg, timer_fisher = [], [], []
        relief_acc_agg, relief_f1_agg, timer_relief = [], [], []
        chi2_acc_agg, chi2_f1_agg, timer_chi2 = [], [], []
        mrmr_acc_agg, mrmr_f1_agg, timer_mrmr = [], [], []
        for kfold_iter in range(1, config['kfolds'] + 1):
            final_kf_iter = kfold_iter == config['kfolds']
            train_set, val_set = kfolds_split(data, kfold_iter, n_splits=config['kfolds'], random_state=config['random_state'])

            # Storing the results we've calculated earlier for all_features
            if final_kf_iter:
                acc_result = round(lists_avg(all_features_acc_agg)*100, 2)
                logger.info(f"all_features accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, 'all_features', all_features_acc_agg, all_features_f1_agg, classes, workdir, timer_all_features)

            # Random Features
            with Timer() as timer:
                random_acc_agg, random_f1_agg = random_features_predict(train_set, val_set, k, all_features, random_acc_agg, random_f1_agg, config['random_state'])
            timer_random.append(timer)
            if final_kf_iter:
                acc_result = round(lists_avg(random_acc_agg) * 100, 2)
                logger.info(f"random_features accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, 'random_features', random_acc_agg, random_f1_agg, classes, workdir, timer_random)

            # Fisher Score Features
            with Timer() as timer:
                fisher_acc_agg, fisher_f1_agg = fisher_ranks_predict(train_set, val_set, k, all_features, fisher_acc_agg, fisher_f1_agg)
            timer_fisher.append(timer)
            if final_kf_iter:
                acc_result = round(lists_avg(fisher_acc_agg) * 100, 2)
                logger.info(f"fisher_score accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, 'fisher', fisher_acc_agg, fisher_f1_agg, classes, workdir, timer_fisher)

            # ReliefF Features
            with Timer() as timer:
                relief_acc_agg, relief_f1_agg = relieff_predict(train_set, val_set, k, all_features, relief_acc_agg, relief_f1_agg)
            timer_relief.append(timer)
            if final_kf_iter:
                acc_result = round(lists_avg(relief_acc_agg) * 100, 2)
                logger.info(f"Relief accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, 'relief', relief_acc_agg, relief_f1_agg, classes, workdir, timer_relief)

            # Chi Square Features
            with Timer() as timer:
                chi2_acc_agg, chi2_f1_agg = chi_square_predict(train_set, val_set, k, all_features, chi2_acc_agg, chi2_f1_agg)
            timer_chi2.append(timer)
            if final_kf_iter:
                acc_result = round(lists_avg(chi2_acc_agg) * 100, 2)
                logger.info(f"chi_square accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, 'chi_square', chi2_acc_agg, chi2_f1_agg, classes, workdir, timer_chi2)

            # mRMR Features
            with Timer() as timer:
                mrmr_acc_agg, mrmr_f1_agg = mrmr_predict(train_set, val_set, k, all_features, mrmr_acc_agg, mrmr_f1_agg)
            timer_mrmr.append(timer)
            if final_kf_iter:
                acc_result = round(lists_avg(mrmr_acc_agg) * 100, 2)
                logger.info(f"mRMR accuracy result: {acc_result}%")
                store_results(config['dataset_name'], feature_percentage, 'mrmr', mrmr_acc_agg, mrmr_f1_agg, classes, workdir, timer_mrmr)

    # TauTransformer Features
    tausformer_main(config, dm_params, [dataset])

    t_test(config['dataset_name'])


def main():
    config = {
        'kfolds': 5,
        'features_percentage': [0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
        'dist_functions': ['wasserstein', 'jm', 'hellinger'],
        'nrows': 10000,
        'features_to_eliminate_prc': [0.0, 0.2, 0.35, 0.5],
        'verbose': False,
        'random_state': 0,
    }

    dm_params = {
        'dim': 2,
        'alpha': 1,
        'eps_type': 'maxmin',
        'epsilon_factor': [10, 100]
    }

    # tuples of datasets names and target column name
    datasets = [
        ('adware_balanced', 'label'), ('ml_multiclass_classification_data', 'target'), ('digits', 'label'),
        ('isolet', 'label'), ('otto_balanced', 'target'), ('gene_data', 'label')
    ]
    datasets = [
        ('adware_balanced', 'label'), ('ml_multiclass_classification_data', 'target'), ('digits', 'label'),
        ('isolet', 'label'), ('otto_balanced', 'target')
    ]
    # config['features_percentage'] = [0.02, 0.05, 0.1, 0.2, 0.3]
    # config['features_to_eliminate_prc'] = [0.0, 0.2, 0.35, 0.5]

    for dataset, label in datasets:
        config['dataset_name'] = dataset
        config['label_column'] = label
        run_experiments(config, dm_params, dataset=(dataset, label))

    all_results_colorful()


if __name__ == '__main__':
    main()
