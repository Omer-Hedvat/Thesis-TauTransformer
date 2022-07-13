from datetime import datetime
from functools import reduce
import itertools
import logging
from math import sqrt
import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from skfeature.function.similarity_based import fisher_score
from ReliefF import ReliefF
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy import stats

from utils.diffusion_maps import diffusion_mapping
from utils.distances import wasserstein_dist, bhattacharyya_dist, hellinger_dist, jm_dist
from utils.files import create_work_dir, read_from_csv, print_separation_dots
from utils.general import flatten, setup_logger
from utils.machine_learning import min_max_scaler

logger = logging.getLogger(__name__)



def kfolds_split(data, iter, n_splits=5, random_state=0):
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True).copy()
    split_len = int(data.shape[0]/n_splits)
    val_i = n_splits - iter
    val_set = data.iloc[val_i*split_len:(val_i+1)*split_len]
    train_set = data[~data.index.isin(val_set.index)]
    return train_set, val_set




def main():
    config = {
        'features_percentage': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'dist_functions': ['wasserstein', 'jm', 'hellinger'],
        'nrows': 500,
        'features_to_reduce_prc': [0.0, 0.2, 0.35, 0.5],
        'dm_dim': [2],
        'alpha': 1,
        'eps_type': 'maxmin',
        'eps_factor': 25
    }

    # tuples of datasets names and target column name
    datasets = [
        ('adware_balanced', 'label'), ('ml_multiclass_classification_data', 'target'), ('digits', 'label'), ('isolet', 'label'),
        ('otto_balanced', 'target')
    ]

    for dataset, label in datasets:
        config['dataset_name'] = dataset
        config['label_column'] = label

    workdir = os.path.join(f'results', config['dataset_name'])
    create_work_dir(workdir, on_exists='ignore')
    setup_logger("config_files/logger_config.json", os.path.join(workdir, f"{config['dataset_name']}_log_{datetime.now().strftime('%d-%m-%Y')}.txt"))
    dataset_dir = f"data/{config['dataset_name']}.csv"

    logger.info(f'{dataset_dir=}')
    data = read_from_csv(dataset_dir, config)
    all_features = data.columns.drop('label')
    classes = list(data['label'].unique())
    logger.info(f"DATA STATS:\ndata shape of {data.shape}\nLabel distributes:\n{data['label'].value_counts().sort_index()}\n")
    n_splits = 5
    for i in range(1, n_splits+1):
        kfolds_split(data, i, n_splits=5, random_state=0)


if __name__ == '__main__':
    main()