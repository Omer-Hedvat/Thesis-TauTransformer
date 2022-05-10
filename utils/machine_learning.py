import logging

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def min_max_scaler(df1, features, df2=None, return_as_df=True):
    """
    activates min_max_scaler over a df and returns the normalized DataFrame
    :param df1: pandas DataFrame which we want to fit_transfomr on
    :param features: a list of columns which are the features
    :param df2: pandas DataFrame - an optional dataframe which we transform only
    :param return_as_df: a boolean flag which determines if we want Numpy array or Pandas DataFrame
    :return: normalized dataframe/s (features only)
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df1_norm = scaler.fit_transform(df1[features])
    if return_as_df:
        df1_norm = pd.DataFrame(df1_norm, columns=features)
    if df2 is not None:
        df2_norm = scaler.transform(df2[features])
        if return_as_df:
            df2_norm = pd.DataFrame(df2_norm, columns=features)
        return df1_norm, df2_norm
    return df1_norm


def export_heatmaps(df, features, dist_type1, dist_type2, to_norm=False):
    import matplotlib as plt
    from main import calc_dist
    from utils.distances import norm_by_dist_type
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