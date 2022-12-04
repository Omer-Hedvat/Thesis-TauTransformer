from joblib import Parallel, delayed
import logging
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from utils.diffusion_maps import diffusion_mapping
from utils.distances import wasserstein_dist, bhattacharyya_dist, hellinger_dist, jm_dist

logger = logging.getLogger(__name__)


class TauTransformer:
    def __init__(
            self, feature_percentage, features_to_eliminate_prc, dist_functions, dm_dim=2, min_feature_std=0, alpha=1,
            eps_type='maxmin', eps_factor=[100, 25], random_state=0, verbose=False
    ):
        self.X = None
        self.y = None
        self.low_std_features = list()
        self.feature_percentage = feature_percentage
        self.features_to_eliminate_prc = features_to_eliminate_prc
        self.dist_functions = dist_functions

        self.dm_dim = dm_dim
        self.min_feature_std = min_feature_std
        self.alpha = alpha
        self.eps_type = eps_type
        self.eps_factor = eps_factor

        self.random_state = random_state
        self.verbose = verbose

        self.all_features = np.array([])
        self.dm_dict = dict()
        self.dists_dict = dict()

        self.k = int()
        self.best_features_idx = list()
        self.best_features = np.array([])

    @staticmethod
    def execute_distance_func(X_arr, y_arr, dist_func_name, feature_idx, cls_feature1, cls_feature2):
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
        assert dist_func_name in ['wasserstein', 'bhattacharyya', 'jm', 'hellinger']
        return {
            'wasserstein': lambda: wasserstein_dist(X_arr, y_arr, feature_idx, cls_feature1, cls_feature2),
            'bhattacharyya': lambda: bhattacharyya_dist(X_arr, y_arr, feature_idx, cls_feature1, cls_feature2),
            'hellinger': lambda: hellinger_dist(X_arr, y_arr, feature_idx, cls_feature1, cls_feature2),
            'jm': lambda: jm_dist(X_arr, y_arr, feature_idx, cls_feature1, cls_feature2)
        }[dist_func_name]()

    @staticmethod
    def flatten(t):
        """
        given a matrix, returns a flatten list
        :param t:
        :return:
        """
        return [item for sublist in t for item in sublist]

    @staticmethod
    def percentage_calculator(array, prc):
        return int(len(array) * prc)

    def drop_low_std_features(self):
        stds = self.X.std(axis=0)
        low_std_feature_indexes = np.where(stds <= self.min_feature_std)[0].tolist()
        self.low_std_features += list(self.all_features[low_std_feature_indexes])
        self.all_features = np.delete(self.all_features, low_std_feature_indexes)
        self.X = np.delete(self.X, low_std_feature_indexes, axis=1)

    def calc_dist(self, dist_func_name):
        """
        Calculates distances of each feature w/ itself in different target classses
        for each DataFrame & distance functions
        :param dist_func_name: Distance function name
        return: df_dists, dist_dict
        df_dists - a flattened df of all features (each feature is a row)
        dist_dict - a dictionary of feature names & dataframes (e.g. {'feature_1': feature_1_df, ...}
        """
        distances = []
        classes = np.unique(self.y)
        for feature_idx in range(len(self.all_features)):
            class_dist = []
            for class_idx in range(len(classes)):
                cls_feature1 = classes[class_idx]
                class_row = [
                    self.execute_distance_func(self.X, self.y, dist_func_name, feature_idx, cls_feature1, cls_feature2)
                    for cls_feature2 in classes[class_idx + 1:]
                ]
                class_dist.append(class_row)
            distances.append(class_dist)

        two_dim_matrix = [self.flatten(dist_mat) for dist_mat in distances]
        dists_arr = np.array([np.array(row) for row in two_dim_matrix])
        dists_dict = {dist_func_name: dists_arr}
        # dist_dict = {f'feature_{class_idx + 1}': pd.DataFrame(mat) for class_idx, mat in enumerate(distances)}
        return dists_dict

    def features_elimination(self):
        """
        A heuristic function for feature elimination.
        the function calls dist_features_elimination() for each distance function which calculates the mean value
        for each feature and returns the feature indexes to eliminate.
        After all results comes back from for each distance function() we combine the results and eliminate the selected features
        """
        features_to_eliminate_df = pd.DataFrame(
            {'features': [*range(0, len(self.all_features), 1)], 'count': len(self.all_features) * [0]}
        )
        for dist, dist_arr in self.dists_dict.items():
            features_to_keep_idx, features_to_eliminate_idx = self.dist_features_elimination(dist_arr)
            for feature in features_to_eliminate_idx:
                features_to_eliminate_df.at[feature, 'count'] += 1

        number_to_eliminate = self.percentage_calculator(self.all_features, self.features_to_eliminate_prc)
        features_to_eliminate_df.sort_values(by='count', ascending=False, inplace=True)
        final_features_to_eliminate_idx = set(features_to_eliminate_df.iloc[:number_to_eliminate]['features'].tolist())
        final_features_to_keep_idx = list(set(range(len(self.all_features))).difference(final_features_to_eliminate_idx))
        final_dists_dict = {key: value[final_features_to_keep_idx] for key, value in self.dists_dict.items()}

        if self.verbose:
            logger.info(
                f"""features_elimination() -  By a majority of votes, a {self.features_to_eliminate_prc * 100}% of the features has been eliminated. 
                The eliminated features are:\n{self.all_features[list(final_features_to_eliminate_idx)]}""")
        return final_dists_dict, final_features_to_keep_idx

    def dist_features_elimination(self, dist_arr):
        """
        Calculates the mean value for every row(feature distances) and returns the best X features to eliminate
        :param dist_arr: a distance matrix MXC^2
        """
        arr_avg = dist_arr.mean(axis=1)
        num_features_to_eliminate = self.percentage_calculator(dist_arr, self.features_to_eliminate_prc)
        features_to_keep_idx = np.sort(np.argsort(arr_avg)[num_features_to_eliminate:])
        features_to_eliminate_idx = list(set(range(len(self.all_features))).difference(features_to_keep_idx))
        return features_to_keep_idx, features_to_eliminate_idx

    def return_best_features_by_kmeans(self, coordinates):
        """
        runs K-means algorithm over the coordinates and returns the best features.
        In each centroid we pick the feature with the smallest value in the X axis (the first axis in coordinates)
        :param coordinates: a 2-dim array with the coordinates from the diffusion maps
        :return: 3 lists. 'best_features_idx' - best features indexes list, 'labels' - the K-means labels
        & 'features_rank' - the features ranked by the smallest value of coordinates first axis
        """
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

    def fit(self, X, y):
        self.all_features = np.append(self.all_features, X.columns)
        self.X = np.asarray(X)
        self.y = np.asarray(y)

        self.drop_low_std_features()

        if self.verbose:
            logger.info(f"Calculating distances for {', '.join(self.dist_functions)}")
        dist_dict = Parallel(n_jobs=len(self.dist_functions))(
            delayed(self.calc_dist)(dist)
            for dist in self.dist_functions
        )
        self.dists_dict = {k: v for x in dist_dict for k, v in x.items()}

        if self.verbose:
            logger.info(f"Eliminating {int(self.features_to_eliminate_prc * 100)}% features using 'features_elimination()' heuristic")

        self.k = self.percentage_calculator(self.all_features, self.feature_percentage)
        if self.features_to_eliminate_prc > 0:
            distances_dict, features_to_keep_idx = self.features_elimination()
        else:
            distances_dict = self.dists_dict.copy()

        if self.verbose:
            logger.info(f"Calculating diffusion maps over the distance matrix")
        dm_results = Parallel(n_jobs=len(self.dist_functions))(
            delayed(diffusion_mapping)(distances_dict[dist], self.alpha, self.eps_type, self.eps_factor[0], dim=self.dm_dim)
            for dist in self.dist_functions
        )
        self.dm_dict = {k: v for k, v in zip(self.dist_functions, dm_results)}

        if self.verbose:
            logger.info(
                f"""Ranking the {int((1 - self.features_to_eliminate_prc) * 100)}% remain features using a combined coordinate matrix ('agg_corrdinates'), 
                inserting 'agg_corrdinates' into a 2nd diffusion map and storing the 2nd diffusion map results into 'final_coordinates'"""
            )

        agg_coordinates = np.concatenate([val['coordinates'] for val in self.dm_dict.values()]).T
        final_dm_results = diffusion_mapping(agg_coordinates, self.alpha, self.eps_type, self.eps_factor[1], dim=self.dm_dim) if len(self.dist_functions) > 1 else agg_coordinates
        self.best_features_idx, labels, features_rank = self.return_best_features_by_kmeans(final_dm_results['coordinates'])
        self.best_features = np.append(self.best_features, self.all_features)
        if self.verbose:
            logger.info(f'Best features by KMeans are: {self.best_features}')
            logger.info(f"Using KMeans algorithm in order to rank the features who are in final_coordinates")

    def transform(self, X):
        self.X = X
        return self.X[self.best_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
