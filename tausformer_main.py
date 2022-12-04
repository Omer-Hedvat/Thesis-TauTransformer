from datetime import datetime
import itertools
import logging
import os

from TauTransformer import TauTransformer
from utils.files import create_work_dir, read_from_csv, print_separation_dots, store_results, all_results_colorful, generate_and_save_scatter_plots
from utils.general import setup_logger, lists_avg, calc_k, arrange_data_features
from utils.machine_learning import t_test, kfolds_split, min_max_scaler
from utils.machine_learning import predict
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

    for feature_percentage, dm_dim in list(itertools.product(config['features_percentage'], config['dm_dim'])):
        api_params['dm_dim'] = dm_dim
        k = calc_k(all_features, feature_percentage)
        if k < 1 or k == len(all_features):
            continue
        logger.info(f"""
        Running over features percentage of {feature_percentage}, which is {k} features out of {data.shape[1] - 1}, with diffusion mapping dimension of {dm_dim}"""
                    )

        for features_to_reduce_prc in config['features_to_reduce_prc']:
            if feature_percentage + features_to_reduce_prc >= 1:
                continue
            print_separation_dots(f'features to reduce heuristic of {int(features_to_reduce_prc*100)}%')

            data_norm = min_max_scaler(data, all_features, return_as_df=True)
            kmeans_acc_agg, kmeans_f1_agg, timer_tau_trans = [], [], []
            for kfold_iter in range(1, config['kfolds'] + 1):
                final_kf_iter = kfold_iter == config['kfolds']
                train_set, val_set = kfolds_split(data_norm, kfold_iter, n_splits=config['kfolds'], random_state=0)
                X_train, y_train, X_test, y_test = arrange_data_features(train_set, val_set, all_features)

                with Timer() as timer:
                    tt = TauTransformer(feature_percentage, features_to_reduce_prc, config['dist_functions'], **api_params)
                    X_tr = tt.fit_transform(X_train, y_train)
                    X_tst = tt.transform(X_test)
                    kmeans_acc, kmeans_f1 = predict(X_tr, y_train, X_tst, y_test)

                kmeans_acc_agg.append(kmeans_acc)
                kmeans_f1_agg.append(kmeans_f1)
                timer_tau_trans.append(timer)
                generate_and_save_scatter_plots(tt.dm_dict, workdir)
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
        'nrows': 100,
        'features_to_eliminate_prc': [0.0, 0.2, 0.35, 0.5],
        'dm_dim': [2],
        'alpha': 1,
        'eps_type': 'maxmin',
        'eps_factor': [10, 100],
        'verbose': False,
        'random_state': 0
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
        ('adware_balanced', 'label'), ('ml_multiclass_classification_data', 'target'), ('digits', 'label'),
        ('isolet', 'label'), ('otto_balanced', 'target'), ('gene_data', 'label')
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