import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import utils


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
    assert function_name in ['wasserstein_dist', 'bhattacharyya_dist', 'jm_dist', 'hellinger_dist']
    return {
        'wasserstein_dist': lambda: utils.wasserstein_dist(df, feature, label1, label2),
        'bhattacharyya_dist': lambda: utils.bhattacharyya_dist(df, feature, label1, label2),
        'jensen_shannon_dist': lambda: utils.jensen_shannon_dist(df, feature, label1, label2),
        'hellinger_dist': lambda: utils.hellinger_dist(df, feature, label1, label2),
        'jm_dist': lambda: utils.jm_dist(df, feature, label1, label2)
    }[function_name]()


def calc_dist(dist_func_name, df, target_col):
    """
    Calculates distances of each feature w/ itself in different target classses
    for each DataFrame & distance functions

    return: df_dists, dist_dict
    df_dists - a flatten df of all features (each feature is a row)
    dist_dict - a dictionary of feature names & dataframes (e.g. {'feature_1': feature_1_df, ...}
    """
    features = df.columns.drop(target_col)
    classes = df[target_col].unique()
    distances = []
    for feature in features:
        class_dist = []
        for cls_feature1 in classes:
            class_row = [execute_distance_func(df, dist_func_name, feature, cls_feature1, cls_feature2) for cls_feature2 in classes]
            class_dist.append(class_row)
        distances.append(class_dist)

    two_d_mat = [utils.flatten(distances[idx]) for idx in range(len(distances))]
    df_dists = pd.DataFrame(two_d_mat)
    dist_dict = {f'feature_{idx + 1}': pd.DataFrame(mat) for idx, mat in enumerate(distances)}
    return df_dists, dist_dict


def export_heatmaps(df, features, dist_type1, dist_type2, to_norm):
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
            feature_dist_mat1 = utils.norm_by_dist_type(feature_mat1)
            feature_dist_mat2 = utils.norm_by_dist_type(feature_mat2)
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


def main():
    glass_df = pd.read_csv('data/glass.csv')


if __name__ == '__main__':
    main()
