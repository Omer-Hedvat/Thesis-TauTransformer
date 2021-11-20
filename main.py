from math import exp, sqrt, log
import matplotlib as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dictances import cosine
from dictances import bhattacharyya
from dictances import bhattacharyya_coefficient
from dictances import jensen_shannon
from dictances import kullback_leibler
from scipy.stats import wasserstein_distance


def main():
    glass_df = pd.read_csv('data/glass.csv')

    def hellinger(p, q):
        """Hellinger distance between two discrete distributions.
           Same as original version but without list comprehension
        """
        # Calculate the square of the difference of ith distr elements
        list_of_squares = [((sqrt(p_i) - sqrt(q_i)) ** 2) for p_i, q_i in zip(p, q)]
        sosq = sum(list_of_squares)
        return sosq / sqrt(2)

    def wasserstein_dist(df, feature, label1, label2):
        dist = wasserstein_distance(df.loc[df['label'] == label1, feature], df.loc[df['label'] == label2, feature])
        return dist

    def bhattacharyya_dist(df, feature, label1, label2):
        dist = bhattacharyya(df.loc[df['label'] == label1, feature], df.loc[df['label'] == label2, feature])
        #     dist= bhattacharyya_distance(df.loc[df['label']==label1, feature], df.loc[df['label']==label2, feature])
        return dist

    def jensen_shannon_dist(df, feature, label1, label2):
        dist = jensen_shannon(df.loc[df['label'] == label1, feature], df.loc[df['label'] == label2, feature])
        return dist

    def hellinger_dist(df, feature, label1, label2):
        dist = hellinger(df.loc[df['label'] == label1, feature], df.loc[df['label'] == label2, feature])
        return dist

    def flatten(t):
        return [item for sublist in t for item in sublist]

    def execute_distance_func(df, function_name, feature, label1, label2):
        """
        Executes various distance function by 'function_name' argument.
        The function calculates the distance between 2 vectors (df column), the vectors are values from the same column but w. different label values.
        by each function_name this function knows to call the right distance function
        :param df: Pandas DataFrame
        :param function_name: the name of the function
        :param feature: the name of the feature/column we want to use the distance on
        :param label1: value of label # 1
        :param label2: value of label # 2
        :return: distance value between the vectors
        """
        assert function_name in ['wasserstein_dist', 'bhattacharyya_dist', 'jensen_shannon_dist', 'hellinger_dist']
        return {
            'wasserstein_dist': lambda: wasserstein_dist(df, feature, label1, label2),
            'bhattacharyya_dist': lambda: bhattacharyya_dist(df, feature, label1, label2),
            'jensen_shannon_dist': lambda: jensen_shannon_dist(df, feature, label1, label2),
            'hellinger_dist': lambda: hellinger_dist(df, feature, label1, label2)
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

        two_d_mat = [flatten(distances[idx]) for idx in range(len(distances))]
        df_dists = pd.DataFrame(two_d_mat)
        dist_dict = {f'feature_{idx + 1}': pd.DataFrame(mat) for idx, mat in enumerate(distances)}
        return df_dists, dist_dict

    def calc_mean_std(df):
        """
        Calculates matrix's mean & std (of entire matrix)
        :return: mean, std
        """
        mean = df.mean().mean()
        var = sum([((x - mean) ** 2) for x in flatten(df.values)]) / len(flatten(df.values))
        std = var ** 0.5
        return mean, std

    def norm_by_dist_type(df, feature, dist_type):
        assert dist_type in ('wasserstein_dist', 'bhattacharyya_dist', 'jensen_shannon_dist', 'hellinger_dist', 'jm_dist')
        _, dist_dict = calc_dist(dist_type, df, 'label')
        feature_mat = dist_dict[feature]
        mean, std = calc_mean_std(feature_mat)
        norm_feature_mat = (feature_mat - mean) / std
        return norm_feature_mat

    def export_plots(df, feature, dist_type1, dist_type2):
        print(f'{feature=}')
        norm_feature_mat1 = norm_by_dist_type(df, feature, dist_type1)
        norm_feature_mat2 = norm_by_dist_type(df, feature, dist_type2)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(6)
        fig.set_figwidth(12)

        sns.heatmap(norm_feature_mat1, annot=True, linewidths=.5, ax=ax1).set(title=dist_type1)
        sns.heatmap(norm_feature_mat2, annot=True, linewidths=.5, ax=ax2).set(title=dist_type2)

        plt.show()


if __name__ == '__main__':
    main()