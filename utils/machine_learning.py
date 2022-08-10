import logging
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold

from utils.general import arrange_data_features

logger = logging.getLogger(__name__)


def min_max_scaler(arr1, features, arr2=None, return_as_df=False):
    """
    activates min_max_scaler over a df and returns the normalized DataFrame
    :param arr1: pandas DataFrame which we want to fit_transfomr on
    :param features: a list of columns which are the features
    :param arr2: pandas DataFrame - an optional dataframe which we transform only
    :param return_as_df: a boolean flag which determines if we want Numpy array or Pandas DataFrame
    :return: normalized dataframe/s (features only)
    """
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    arr1_norm = scaler.fit_transform(arr1)
    if return_as_df:
        arr1_norm = pd.DataFrame(arr1_norm, columns=features)
    if arr2 is not None:
        arr2_norm = scaler.transform(arr2)
        if return_as_df:
            arr2_norm = pd.DataFrame(arr2_norm, columns=features)
        return arr1_norm, arr2_norm
    return arr1_norm


def export_heatmaps(df, features, dist_type1, dist_type2, to_norm=False):
    import matplotlib as plt
    from utils.distances import norm_by_dist_type, calc_dist
    import seaborn as sns

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


def kfolds_split(data, iter, n_splits=5, random_state=0):
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True).copy()
    split_len = int(data.shape[0]/n_splits)
    val_i = n_splits - iter
    val_set = data.iloc[val_i*split_len:(val_i+1)*split_len]
    train_set = data[~data.index.isin(val_set.index)]
    return train_set, val_set


def calc_f1_score(f1_lists):
    return list(np.array(f1_lists).mean(axis=0))


def t_test(dataset_name):
    """
    :param dataset_name: the name of the dataset we are using
    :return: add all t_test p-valus for the dataset.
    The T test calculation is done for each of our methods versus the rest of our conventional
    methods we compare: 'random_features', 'fisher', 'relief', 'chi_square'
    """
    from scipy import stats

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


def predict(X_train, y_train, X_val, y_val):
    clf = RandomForestClassifier(random_state=1)
    multi_target_forest = OneVsRestClassifier(clf, n_jobs=-1)
    model = multi_target_forest.fit(X_train, y_train)
    validation_preds = model.predict(X_val)

    train_acc = metrics.accuracy_score(y_val, validation_preds)
    f1_scores_list = metrics.f1_score(y_val, validation_preds, average=None)
    return train_acc, f1_scores_list


def random_features_predict(train_set, val_set, k, all_features, random_acc_agg, random_f1_agg, random_state):
    import random

    random.seed(random_state)
    random_features = random.sample(list(all_features), k)
    X_tr, y_tr, X_test, y_test = arrange_data_features(train_set, val_set, random_features, return_y=True)
    random_acc, random_f1 = predict(X_tr, y_tr, X_test, y_test)
    random_acc_agg.append(random_acc)
    random_f1_agg.append(random_f1)

    return random_acc_agg, random_f1_agg


def fisher_ranks_predict(train_set, val_set, k, all_features, fisher_acc_agg, fisher_f1_agg):
    from skfeature.function.similarity_based import fisher_score

    fisher_ranks = fisher_score.fisher_score(train_set[all_features].to_numpy(), train_set['label'].to_numpy())
    fisher_features_idx = np.argsort(fisher_ranks, 0)[::-1][:k]
    fisher_features = all_features[fisher_features_idx]
    X_tr, y_tr, X_test, y_test = arrange_data_features(train_set, val_set, fisher_features, return_y=True)

    fisher_acc, fisher_f1 = predict(X_tr, y_tr, X_test, y_test)
    fisher_acc_agg.append(fisher_acc)
    fisher_f1_agg.append(fisher_f1)
    return fisher_acc_agg, fisher_f1_agg


def relieff_predict(train_set, val_set, k, all_features, relief_acc_agg, relief_f1_agg):
    from ReliefF import ReliefF

    fs = ReliefF(n_neighbors=1, n_features_to_keep=k)
    X_tr, y_tr, X_test, y_test = arrange_data_features(train_set, val_set, all_features, return_y=True)
    X_tr = fs.fit_transform(train_set[all_features].to_numpy(), train_set['label'].to_numpy())
    X_test = fs.transform(val_set[all_features].to_numpy())

    relief_acc, relief_f1 = predict(X_tr, y_tr, X_test, y_test)
    relief_acc_agg.append(relief_acc)
    relief_f1_agg.append(relief_f1)
    return relief_acc_agg, relief_f1_agg


def chi_square_predict(train_set, val_set, k, all_features, chi_square_acc_agg, chi_square_f1_agg):
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import SelectKBest

    chi_features = SelectKBest(chi2, k=k)
    X_tr, y_tr, X_test, y_test = arrange_data_features(train_set, val_set, all_features, return_y=True)
    X_tr_norm, X_test_norm = min_max_scaler(train_set, all_features, val_set, return_as_df=False)
    X_tr = chi_features.fit_transform(X_tr_norm, y_tr)
    X_test = chi_features.transform(X_test_norm)

    chi2_acc, chi2_f1 = predict(X_tr, y_tr, X_test, y_test)
    chi_square_acc_agg.append(chi2_acc)
    chi_square_f1_agg.append(chi2_f1)
    return chi_square_acc_agg, chi_square_f1_agg


def mrmr_predict(train_set, val_set, k, all_features, mrmr_acc_agg, mrmr_f1_agg):
    from mrmr import mrmr_classif

    mrmr_features = mrmr_classif(X=train_set[all_features], y=train_set['label'], K=k)
    X_tr, y_tr, X_test, y_test = arrange_data_features(train_set, val_set, mrmr_features, return_y=True)

    mrmr_acc, mrmr_f1 = predict(X_tr, y_tr, X_test, y_test)
    mrmr_acc_agg.append(mrmr_acc)
    mrmr_f1_agg.append(mrmr_f1)
    return mrmr_acc_agg, mrmr_f1_agg


def return_best_features_by_kmeans(coordinates, k):
    from sklearn.cluster import KMeans

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