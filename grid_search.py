import pandas as pd
from TauTransformer import TauTransformer
from utils.files import read_from_csv
import itertools
from utils.general import lists_avg, arrange_data_features, percentage_calculator
from utils.machine_learning import kfolds_split, min_max_scaler
from utils.machine_learning import predict
from utils.timer import Timer


def grid_search(config):
    dataset = config['dataset_name']
    features_percentages = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

    dm_params = {
        'dim': 2,
        'alpha': 1,
        'eps_type': 'maxmin'
    }

    grid_search_dict = {
        'features_to_eliminate_prc': [0.0, 0.2, 0.35, 0.5],
        'epsilon_factor': [[25, 25], [100, 10], [50, 50], [10, 100]],
        'dist_functions': ['wasserstein', 'jm', 'hellinger', ['wasserstein', 'jm'], ['wasserstein', 'hellinger'], ['jm', 'hellinger'], ['wasserstein', 'jm', 'hellinger']]
    }

    keys, values = zip(*grid_search_dict.items())
    permutations_list_of_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    dataset_dir = f"data/{config['dataset_name']}.csv"
    data, config['dataset_name'] = read_from_csv(dataset_dir, config)
    all_features = data.columns.drop('label')
    data_norm = min_max_scaler(data, all_features, return_as_df=True)
    kmeans_acc_agg,  timer_list = [], []

    results_list = list()
    for feature_prc in features_percentages:
        print(f"{feature_prc=}")
        k = percentage_calculator(feature_prc, array=all_features)
        if k < 1 or k == len(all_features):
            continue

        for idx, permutation in enumerate(permutations_list_of_dicts):
            print(f"{idx=}, {permutation=}")
            dm_params['epsilon_factor'] = permutation['epsilon_factor']

            for kfold_iter in range(1, config['kfolds'] + 1):
                train_set, val_set = kfolds_split(data_norm, kfold_iter, n_splits=config['kfolds'], random_state=0)
                X_train, y_train, X_test, y_test = arrange_data_features(train_set, val_set, all_features)

                with Timer() as timer:
                    tt = TauTransformer(feature_prc, permutation['features_to_eliminate_prc'], permutation['dist_functions'], dm_params)
                    X_tr = tt.fit_transform(X_train, y_train)
                    X_tst = tt.transform(X_test)
                    kmeans_acc, kmeans_f1 = predict(X_tr, y_train, X_tst, y_test)

                kmeans_acc_agg.append(kmeans_acc)
                timer_list.append(timer)

            acc_result = round(lists_avg(kmeans_acc_agg), 6)
            timer_avg = round(lists_avg([t.to_int() for t in timer_list]), 6)
            results_list.append([feature_prc, *list(permutation.values()), acc_result, timer_avg])

    results_df = (
        pd.DataFrame(
            results_list, columns=['feature_prc', 'features_elimination_prc', 'epsilon_factor', 'dist_functions', 'accuracy', 'round_time'])
        .sort_values(by=['feature_prc', 'accuracy'], ascending=[True, False])
    )

    results_df.to_csv(f'results/grid_search/{dataset}.csv', index=False)


def main():
    config = {'kfolds': 5, 'nrows': 10000, 'verbose': False, 'random_state': 0}

    datasets = [
        ('adware_balanced', 'label'), ('ml_multiclass_classification_data', 'target'), ('digits', 'label'),
        ('isolet', 'label'), ('otto_balanced', 'target')
    ]

    for dataset, label in datasets:
        print(f"{dataset=}")
        config['dataset_name'] = dataset
        config['label_column'] = label
        grid_search(config)


if __name__ == '__main__':
    main()