import logging
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from utils.diffusion_maps import diffusion_mapping
from utils.distances import wasserstein_dist, bhattacharyya_dist, hellinger_dist, jm_dist
from utils.machine_learning import min_max_scaler

logger = logging.getLogger(__name__)


class TauTransformer:
    def __init__(
            self, X, y, feature_percentage, features_to_reduce_prc, dist_functions, dm_dim=2, alpha=1, eps_type='maxmin', eps_factor=25,
            random_state=0, verbose=False
    ):
        self.X = X
        self.y = y
        self.feature_percentage = feature_percentage
        self.features_to_reduce_prc = features_to_reduce_prc
        self.dist_functions = dist_functions

        self.dm_dim = dm_dim
        self.alpha = alpha
        self.eps_type = eps_type
        self.eps_factor = eps_factor

        self.random_state = random_state
        self.verbose = verbose

        self.all_features = self.X.columns
        self.dm_dict = dict()
        self.df_dists = None
        self.k = self.calc_k(self.all_features, self.feature_percentage)

    @staticmethod
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
        assert function_name in ['wasserstein', 'bhattacharyya', 'jm', 'hellinger']
        return {
            'wasserstein': lambda: wasserstein_dist(df, feature, label1, label2),
            'bhattacharyya': lambda: bhattacharyya_dist(df, feature, label1, label2),
            'hellinger': lambda: hellinger_dist(df, feature, label1, label2),
            'jm': lambda: jm_dist(df, feature, label1, label2)
        }[function_name]()

    @staticmethod
    def flatten(t):
        """
        given a matrix, returns a flatten list
        :param t:
        :return:
        """
        return [item for sublist in t for item in sublist]

    @staticmethod
    def calc_k(features, prc):
        return int(len(features) * prc)

    def calc_dist(self, dist_func_name, X_tr_norm, label_col_name):
        """
        Calculates distances of each feature w/ itself in different target classses
        for each DataFrame & distance functions
        :param dist_func_name: Distance function name
        :param X_tr_norm: the normalized X DF
        :param label_col_name: label column name in the data
        return: df_dists, dist_dict
        df_dists - a flatten df of all features (each feature is a row)
        dist_dict - a dictionary of feature names & dataframes (e.g. {'feature_1': feature_1_df, ...}
        """
        features = self.X.columns
        self.y.reset_index(drop=True, inplace=True)
        X_tr_norm[label_col_name] = self.y
        distances = []
        for feature in features:
            class_dist = []
            for cls_feature1 in self.y.unique():
                class_row = [
                    self.execute_distance_func(X_tr_norm, dist_func_name, feature, cls_feature1, cls_feature2)
                    if cls_feature1 != cls_feature2 else 0
                    for cls_feature2 in self.y.unique()
                ]
                class_dist.append(class_row)
            distances.append(class_dist)

        two_d_mat = [self.flatten(distances[idx]) for idx in range(len(distances))]
        self.df_dists = pd.DataFrame(two_d_mat)
        dist_dict = {f'feature_{idx + 1}': pd.DataFrame(mat) for idx, mat in enumerate(distances)}
        return dist_dict

    def features_reduction(self, dists_dict):
        features_to_reduce_df = pd.DataFrame(
            {'features': [*range(0, len(self.all_features), 1)], 'count': len(self.all_features) * [0]})
        for dist, self.df_dists in dists_dict.items():
            features_to_keep_idx, features_to_reduce_idx = self.dist_features_reduction()
            for feature in features_to_reduce_idx:
                features_to_reduce_df.at[feature, 'count'] += 1

        features_to_reduce_df.sort_values(by='count', ascending=False, inplace=True)
        final_features_to_reduce_idx = set(features_to_reduce_df.iloc[:self.k]['features'].tolist())
        final_features_to_keep_idx = list(set(self.df_dists.index).difference(final_features_to_reduce_idx))
        final_dists_dict = {key: value.iloc[final_features_to_keep_idx] for key, value in dists_dict.items()}

        if self.verbose:
            logger.info(
                f"""features_reduction() -  By a majority of votes, a {self.features_to_reduce_prc}%, {self.k} 
                features reduction of features has been to:\n{self.all_features[list(final_features_to_reduce_idx)]}""")
        return final_dists_dict, final_features_to_keep_idx

    def dist_features_reduction(self):
        df_feature_avg = self.df_dists.mean(axis=1)
        num_features_to_reduce = int(len(self.df_dists) * self.features_to_reduce_prc)
        features_to_keep_idx = df_feature_avg.iloc[np.argsort(df_feature_avg)][
                               num_features_to_reduce:].index.sort_values()
        features_to_reduce_idx = list(set(df_feature_avg.index).difference(features_to_keep_idx))
        return features_to_keep_idx, features_to_reduce_idx

    def return_best_features_by_kmeans(self, coordinates):
        features_rank = np.argsort(coordinates[0])
        kmeans = KMeans(n_clusters=self.k, random_state=self.random_state)
        labels = kmeans.fit(coordinates.T).labels_
        best_features_idx = []
        selected_cetroids = []
        for idx in features_rank:
            if labels[idx] not in selected_cetroids:
                selected_cetroids.append(labels[idx])
                best_features_idx.append(idx)
        return best_features_idx, labels, features_rank

    def transform(self):
        dists_dict = dict()
        if self.verbose:
            logger.info(f"Calculating distances for {', '.join(self.dist_functions)}")
        for dist in self.dist_functions:
            X_tr_norm = min_max_scaler(self.X, self.all_features)
            _ = self.calc_dist(dist, X_tr_norm, 'label')
            dists_dict[dist] = self.df_dists

        if self.verbose:
            logger.info(f"Reducing {int(self.features_to_reduce_prc * 100)}% features using 'features_reduction()' heuristic")

        if self.features_to_reduce_prc > 0:
            distances_dict, features_to_keep_idx = self.features_reduction(dists_dict)
            self.all_features = self.all_features[features_to_keep_idx]
        else:
            distances_dict = dists_dict.copy()

        if self.verbose:
            logger.info(f"Calculating diffusion maps over the distance matrix")
        for dist in self.dist_functions:
            coordinates, ranking = diffusion_mapping(distances_dict[dist], self.alpha, self.eps_type, self.eps_factor, dim=self.dm_dim)
            self.dm_dict[dist] = {'coordinates': coordinates, 'ranking': ranking}

        if self.verbose:
            logger.info(
                f"""Ranking the {int((1 - self.features_to_reduce_prc) * 100)}% remain features using a combined coordinate matrix ('agg_corrdinates'), 
                inserting 'agg_corrdinates' into a 2nd diffusion map and storing the 2nd diffusion map results into 'final_coordinates'"""
            )
        agg_coordinates = np.concatenate([val['coordinates'] for val in self.dm_dict.values()]).T
        final_coordinates, final_ranking = diffusion_mapping(agg_coordinates, self.alpha, self.eps_type, self.eps_factor, dim=self.dm_dim)
        best_features_idx, labels, features_rank = self.return_best_features_by_kmeans(final_coordinates)

        if self.verbose:
            logger.info(f'Best features by KMeans are: {self.all_features[best_features_idx]}')
            logger.info(f"Using KMeans algorithm in order to rank the features who are in final_coordinates")
        return self.all_features[best_features_idx], best_features_idx
