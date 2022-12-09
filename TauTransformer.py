from joblib import Parallel, delayed
import logging
import numpy as np

from sklearn.cluster import KMeans

from utils.diffusion_maps import diffusion_mapping
from utils.distances import wasserstein_dist, bhattacharyya_dist, hellinger_dist, jm_dist
from utils.general import ndarray_to_df_w_index_names, percentage_calculator
logger = logging.getLogger(__name__)


class TauTransformer:
    def __init__(
        self, feature_percentage, features_to_eliminate_prc, dist_functions, dm_params, min_feature_std=0, random_state=0, verbose=False
    ):
        self.X = None
        self.y = None
        self.min_feature_std = min_feature_std
        self.low_std_features = list()
        self.feature_percentage = feature_percentage
        self.features_to_eliminate_prc = features_to_eliminate_prc
        self.dist_functions = dist_functions
        self.features_rank_indexes = list()

        self.dm1_params = dm_params.copy()
        self.dm1_params['epsilon_factor'] = dm_params['epsilon_factor'][0]
        self.dm2_params = dm_params.copy()
        self.dm2_params['epsilon_factor'] = dm_params['epsilon_factor'][1]

        self.random_state = random_state
        self.verbose = verbose

        self.all_features = np.array([])
        self.init_num_of_features = int()
        self.dm_dict = dict()
        self.dists_dict = dict()

        self.k = int()
        self.best_features_idx = list()
        self.best_features = np.array([])
        self.results_dict = dict()

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
        """
        return [item for sublist in t for item in sublist]

    def drop_low_std_features(self):
        stds = self.X.std(axis=0)
        low_std_feature_indexes = np.where(stds <= self.min_feature_std)[0].tolist()
        self.low_std_features += list(self.all_features[low_std_feature_indexes])
        self.all_features = np.delete(self.all_features, low_std_feature_indexes)
        self.X = np.delete(self.X, low_std_feature_indexes, axis=1)
        self.results_dict['low_std_features'] = self.low_std_features

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

    def calculate_features_ranks(self):
        """
        The method calculates all the results from rank_features_by_dist_mean() function and stores them into
        self.features_rank_indexes
        """
        rank_lists = [self.rank_features_by_dist_mean(dist_arr) for dist_arr in self.dists_dict.values()]
        consolidated_feature_ranks = []
        for i in range(len(self.all_features)):
            consolidated_feature_ranks.append(sum([np.where(rank_list == i)[0][0] for rank_list in rank_lists]))
        self.features_rank_indexes = np.argsort(consolidated_feature_ranks)

    @staticmethod
    def rank_features_by_dist_mean(dist_arr):
        """
        Calculates the mean value of every row
        :param dist_arr: distances array
        :return: the sorted indexes on descending order of the mean values
        """
        arr_avg = dist_arr.mean(axis=1)
        sorted_indexes_by_mean_arr = np.argsort(arr_avg)[::-1]
        return sorted_indexes_by_mean_arr

    def features_elimination(self):
        """
        A heuristic function for feature elimination.
        the method relies on the self.consolidate_features_ranks() method rankings.
        The method drops the weakest features from the ranking
        """
        if self.features_to_eliminate_prc == 0:
            return self.dists_dict.copy()

        if self.verbose:
            logger.info(
                f"Eliminating {int(self.features_to_eliminate_prc * 100)}% features using 'features_elimination()' heuristic")

        number_to_eliminate = min(
            percentage_calculator(self.features_to_eliminate_prc, num=self.init_num_of_features),
            self.init_num_of_features - self.k - len(self.low_std_features)
        )
        final_features_to_eliminate_idx = self.features_rank_indexes[-number_to_eliminate:]
        final_features_to_keep_idx = list(
            set(range(len(self.all_features))).difference(final_features_to_eliminate_idx))
        final_dists_dict = {key: value[final_features_to_keep_idx] for key, value in self.dists_dict.items()}
        self.results_dict['eliminated_features'] = self.all_features[list(final_features_to_eliminate_idx)]
        self.all_features = self.all_features[list(final_features_to_keep_idx)]

        if self.verbose:
            logger.info(
                f"""features_elimination() -  By a majority of votes, a {self.features_to_eliminate_prc * 100}% of the features has been eliminated. 
                The eliminated features are:\n{self.all_features[list(final_features_to_eliminate_idx)]}""")
        return final_dists_dict

    def return_best_features_by_kmeans(self, coordinates):
        """
        runs K-means algorithm over the coordinates and returns the best features.
        In each cluster we order the features by their mean rank we calculated earlier and we pick the feature with the
        highest mean rank
        :param coordinates: a 2-dim array with the coordinates from the diffusion maps
        :return: 2 lists. 'best_features_idx' - best features indexes list, 'labels' - the K-means labels
        """
        kmeans = KMeans(n_clusters=self.k, random_state=self.random_state)
        labels = kmeans.fit(coordinates.T).labels_
        num_of_clusters = len(np.unique(labels))

        # Fetch clusters into lists
        feature_index_by_clusters = [[] for _ in range(self.k)]
        for i, centroid in enumerate(labels):
            feature_index_by_clusters[centroid].append(i)

        feature_index_by_clusters = [x for x in feature_index_by_clusters if x]

        # Order clusters by mean rank
        feature_index_by_clusters_ordered = [[] for _ in range(num_of_clusters)]
        feature_names_by_clusters_ordered = [[] for _ in range(num_of_clusters)]
        for cluster_ind, cluster_items in enumerate(feature_index_by_clusters):
            for item in self.features_rank_indexes:
                if item in cluster_items:
                    feature_index_by_clusters_ordered[cluster_ind].append(item)
                    feature_names_by_clusters_ordered[cluster_ind].append(self.all_features[item])

        # A fix in case there aren't enough clusters returned from KMeans
        if num_of_clusters < self.k:
            num_of_missing_clusters = self.k - num_of_clusters
            optional_features = {
                elem: cluster_ind
                for cluster_ind, cluster in enumerate(feature_index_by_clusters_ordered)
                for elem in cluster[1:]
                if len(cluster) > 1
            }
            ordered_new_feature_clusters = [feature for feature in self.features_rank_indexes if feature in optional_features.keys()][:num_of_missing_clusters]
            for feature in ordered_new_feature_clusters:
                cluster_ind = optional_features[feature]
                feature_index_by_clusters_ordered[cluster_ind].remove(feature)
                feature_names_by_clusters_ordered[cluster_ind].remove(self.all_features[feature])
            missing_clusters_ind = [[feature] for feature in ordered_new_feature_clusters]
            feature_index_by_clusters_ordered.extend(missing_clusters_ind)
            feature_names_by_clusters_ordered.extend(self.all_features[[missing_clusters_ind]])

        # Store ordered clusters in results_dict
        self.results_dict['feature_index_by_clusters'] = feature_index_by_clusters_ordered
        self.results_dict['feature_names_by_clusters'] = feature_names_by_clusters_ordered

        # Pick the best feature from every cluster
        best_features_idx = [cluster[0] for cluster in feature_index_by_clusters_ordered]
        return best_features_idx, labels

    def fit(self, X, y):
        self.all_features = np.append(self.all_features, X.columns)
        self.init_num_of_features = len(self.all_features)
        self.X = np.asarray(X)
        self.y = np.asarray(y)

        self.k = percentage_calculator(self.feature_percentage, num=self.init_num_of_features)
        self.results_dict['k'] = self.k

        self.drop_low_std_features()

        if self.verbose:
            logger.info(f"Calculating distances for {', '.join(self.dist_functions)}")
        dist_dict = Parallel(n_jobs=len(self.dist_functions))(
            delayed(self.calc_dist)(dist)
            for dist in self.dist_functions
        )
        self.dists_dict = {k: v for x in dist_dict for k, v in x.items()}
        self.results_dict['distance_matrix'] = {k: ndarray_to_df_w_index_names(v, self.all_features) for k, v in
                                                self.dists_dict.items()}

        self.calculate_features_ranks()

        distances_dict = self.features_elimination()

        if self.verbose:
            logger.info(f"Calculating diffusion maps over the distance matrix")
        dm_results = Parallel(n_jobs=len(self.dist_functions))(
            delayed(diffusion_mapping)(distances_dict[dist], **self.dm1_params) for dist in self.dist_functions
        )
        self.dm_dict = {k: v for k, v in zip(self.dist_functions, dm_results)}

        self.results_dict['dm1'] = {k: ndarray_to_df_w_index_names(v['coordinates'].T, self.all_features) for k, v in
                                    self.dm_dict.items()}

        if self.verbose:
            logger.info(
                f"""Ranking the {int((1 - self.features_to_eliminate_prc) * 100)}% remain features using a combined coordinate matrix ('agg_corrdinates'), 
                inserting 'agg_corrdinates' into a 2nd diffusion map and storing the 2nd diffusion map results into 'final_coordinates'"""
            )

        agg_coordinates = np.concatenate([val['coordinates'] for val in self.dm_dict.values()]).T
        final_dm_results = diffusion_mapping(agg_coordinates, **self.dm2_params) if len(self.dist_functions) > 1 else agg_coordinates
        self.results_dict['dm2'] = ndarray_to_df_w_index_names(final_dm_results['coordinates'].T, self.all_features)

        self.best_features_idx, labels = self.return_best_features_by_kmeans(final_dm_results['coordinates'])
        self.best_features = self.all_features[self.best_features_idx]
        if self.verbose:
            logger.info(f'Best features by KMeans are: {self.best_features}')
            logger.info(f"Using KMeans algorithm in order to rank the features who are in final_coordinates")

    def transform(self, X):
        self.X = X
        return self.X[self.best_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
