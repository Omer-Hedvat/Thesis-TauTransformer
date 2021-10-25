from math import exp, sqrt, log
import numpy as np
import pandas as pd

from dictances import cosine
from dictances import bhattacharyya
from dictances import bhattacharyya_coefficient
from dictances import jensen_shannon
from dictances import kullback_leibler
from scipy.stats import wasserstein_distance

def main():
    df = pd.read_csv('data/glass.csv')

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

    def execute_function(df, function_name, feature, label1, label2):
        return {
            'wasserstein_dist': lambda: wasserstein_dist(df, feature, label1, label2),
            'bhattacharyya_dist': lambda: bhattacharyya_dist(df, feature, label1, label2),
            'jensen_shannon_dist': lambda: jensen_shannon_dist(df, feature, label1, label2),
            'hellinger_dist': lambda: hellinger_dist(df, feature, label1, label2),
            'jm_dist': lambda: jm_dist(df, feature, label1, label2)
        }[function_name]()

    def calc_dist(dist_func_name, df, target_col):
        features = df.columns.drop(target_col)
        classes = df[target_col].unique()
        distances = []
        for feature in features:
            class_dist = []
            for cls_feature1 in classes:
                class_row = [execute_function(df, dist_func_name, feature, cls_feature1, cls_feature2) for cls_feature2 in classes]
                class_dist.append(class_row)
            distances.append(class_dist)

        two_d_mat = [flatten(distances[idx]) for idx in range(len(distances))]
        df_dists = pd.DataFrame(two_d_mat)
        return df_dists


if __name__ == '__main__':
    main()