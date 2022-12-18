from joblib import Parallel, delayed
import logging
import numpy as np

import fast_ml.feature_selection as fast_fs
from sklearn.cluster import KMeans

from utils.diffusion_maps import diffusion_mapping
from utils.distances import wasserstein_dist, bhattacharyya_dist, hellinger_dist, jm_dist
from utils.general import ndarray_to_df, percentage_calculator
logger = logging.getLogger(__name__)


class TauTransformer:
    def __init__(
        self, feature_percentage, features_to_eliminate_prc, dist_functions, dm_params, max_prc_null_values=0.8,
            random_state=0, verbose=False
    ):
        """
        Initiates the class
        :param feature_percentage: Which percentage of the features do we want to return
        :param features_to_eliminate_prc: Which percentage of features do we want to eliminate using the elimination heuristic
        :param dist_functions: a list that describes which distance functions we wish to use (Hellinger/JM/Wasserstein)
        :param dm_params: a dictionary of diffusion maps parameters
        :param max_prc_null_values: a threshold of maximal value of how many null values per column
        :param random_state: Controls the randomness of the algorithm
        :param verbose: Controls the verbosity when fitting and predicting.
        """
        self.X = None
        self.y = None
        self.max_prc_null_values = max_prc_null_values
        self.init_eliminated_features = list()
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

        self.features = np.array([])
        self.init_num_of_features = int()
        self.dm_dict = dict()
        self.dists_dict = dict()

        self.k = int()
        self.best_features_idx = list()
        self.best_features = np.array([])
        self.results_dict = dict()

    def delete_features(self, unwanted_features_names=None, unwanted_features_indexes=None, df=None):
        """
        Gets a list of unwanted featuren names and deletes them from 'self.features' & self.X
        :param unwanted_features_names: a list of feature names
        :param unwanted_features_indexes: a list of feature indexes
        :param df: a df w/ columns names
        return: In case df is not None the method returns a fixed df
        """
        assert unwanted_features_names is not None or unwanted_features_indexes is not None

        unwanted_features_indexes = [ind for ind, feature in enumerate(self.features) if
                                     feature in unwanted_features_names] if unwanted_features_indexes is None else unwanted_features_indexes

        unwanted_features_names = self.features[unwanted_features_indexes] if unwanted_features_names is None else unwanted_features_names

        # Delete from self.features
        self.features = np.delete(self.features, unwanted_features_indexes)

        # Delete from self.X
        self.X = np.delete(self.X, unwanted_features_indexes, axis=1)

        # Delete from DF
        if df is not None:
            df.drop(unwanted_features_names, axis=1, inplace=True)
            return df

    def initial_features_removal(self):
        """
        removes Null & constant features from self.features & self.X
        """
        data = ndarray_to_df(self.X, self.features, axis=1)

        # Drop Null Features
        null_features_list = data.columns[(data.isna().sum() / data.shape[0]) > self.max_prc_null_values].to_list()
        self.init_eliminated_features.extend(null_features_list)
        self.results_dict['null_features'] = null_features_list
        data = self.delete_features(null_features_list, df=data)

        # Drop Constant Features
        constant_features = fast_fs.get_constant_features(data)
        constant_feature_list = constant_features['Var'].to_list()
        self.init_eliminated_features.extend(constant_feature_list)
        self.results_dict['constant_features'] = constant_feature_list
        data = self.delete_features(constant_feature_list, df=data)

    @staticmethod
    def execute_distance_func(X_arr, y_arr, dist_func_name, feature_idx, cls_feature1, cls_feature2):
        """
        Executes various distance functions by 'function_name' argument.
        The function calculates the distance between 2 vectors (df column), the vectors are values from the same column but w. different label values.
        by each function_name this function knows to call the right distance function
        :param X_arr: features numpy ndarray
        :param y_arr: label's ndarray
        :param dist_func_name: the name of the requested distance function
        :param feature_idx: the name of the feature/column we want to use the distance on
        :param cls_feature1: value of label # 1
        :param cls_feature2: value of label # 2
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

    def calc_dist(self, dist_func_name):
        """
        Calculates distances of each feature w/ itself in different target classes
        for each DataFrame & distance functions
        :param dist_func_name: Distance function name
        return: df_dists, dist_dict
        df_dists - a flattened df of all features (each feature is a row)
        dist_dict - a dictionary of feature names & dataframes (e.g. {'feature_1': feature_1_df, ...}
        """
        distances = []
        classes = np.unique(self.y)
        for feature_idx in range(len(self.features)):
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
        for i in range(len(self.features)):
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
        number_to_eliminate = max(
            percentage_calculator(self.features_to_eliminate_prc, num=self.init_num_of_features) - len(
                self.init_eliminated_features) - self.k, 0)
        if number_to_eliminate == 0:
            return self.dists_dict.copy()

        if self.verbose:
            logger.info(
                f"Eliminating {int(self.features_to_eliminate_prc * 100)}% features using 'features_elimination()' heuristic")

        final_features_to_eliminate_idx = self.features_rank_indexes[-number_to_eliminate:]
        final_features_to_keep_idx = list(
            set(range(len(self.features))).difference(final_features_to_eliminate_idx))
        final_dists_dict = {key: value[final_features_to_keep_idx] for key, value in self.dists_dict.items()}
        self.results_dict['eliminated_features'] = self.features[list(final_features_to_eliminate_idx)]

        # self.features = self.features[list(final_features_to_keep_idx)]
        self.delete_features(unwanted_features_indexes=final_features_to_eliminate_idx)

        if self.verbose:
            logger.info(
                f"""features_elimination() -  By a majority of votes, a {self.features_to_eliminate_prc * 100}% of the features has been eliminated. 
                The eliminated features are:\n{self.features[list(final_features_to_eliminate_idx)]}""")
        return final_dists_dict

    def return_best_features_by_kmeans(self, coordinates):
        """
        runs K-means algorithm over the coordinates and returns the best features.
        In each cluster we order the features by their mean rank we calculated earlier, and we pick the feature with the
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
                    feature_names_by_clusters_ordered[cluster_ind].append(self.features[item])

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
                feature_names_by_clusters_ordered[cluster_ind].remove(self.features[feature])
            missing_clusters_ind = [[feature] for feature in ordered_new_feature_clusters]
            feature_index_by_clusters_ordered.extend(missing_clusters_ind)
            feature_names_by_clusters_ordered.extend(self.features[[missing_clusters_ind]])

        # Store ordered clusters in results_dict
        self.results_dict['feature_index_by_clusters'] = feature_index_by_clusters_ordered
        self.results_dict['feature_names_by_clusters'] = feature_names_by_clusters_ordered

        # Pick the best feature from every cluster
        best_features_idx = [cluster[0] for cluster in feature_index_by_clusters_ordered]
        return best_features_idx, labels

    def fit(self, X, y):
        """
        Computes the best k features
        :param X: a scaled numpy matrix
        :param y: an array of labels (int only)
        """
        self.features = np.append(self.features, X.columns)
        self.init_num_of_features = len(self.features)
        self.X = np.asarray(X)
        self.y = np.asarray(y)

        self.k = percentage_calculator(self.feature_percentage, num=self.init_num_of_features)
        self.results_dict['k'] = self.k

        self.initial_features_removal()
        if len(self.features) <= self.k:
            self.best_features = self.features
            return

        if self.verbose:
            logger.info(f"Calculating distances for {', '.join(self.dist_functions)}")
        dist_dict = Parallel(n_jobs=len(self.dist_functions))(
            delayed(self.calc_dist)(dist)
            for dist in self.dist_functions
        )
        self.dists_dict = {k: v for x in dist_dict for k, v in x.items()}
        self.results_dict['distance_matrix'] = {k: ndarray_to_df(v, self.features, axis=0) for k, v in self.dists_dict.items()}

        self.calculate_features_ranks()

        distances_dict = self.features_elimination()

        if self.verbose:
            logger.info(f"Calculating diffusion maps over the distance matrix")
        dm_results = Parallel(n_jobs=len(self.dist_functions))(
            delayed(diffusion_mapping)(distances_dict[dist], **self.dm1_params) for dist in self.dist_functions
        )
        self.dm_dict = {k: v for k, v in zip(self.dist_functions, dm_results)}

        self.results_dict['dm1'] = {
            k: ndarray_to_df(v['coordinates'].T, self.features, axis=0) for k, v in self.dm_dict.items()
        }

        if self.verbose:
            logger.info(
                f"""Ranking the {int((1 - self.features_to_eliminate_prc) * 100)}% remain features using a combined coordinate matrix ('agg_coordinates'), 
                inserting 'agg_coordinates' into a 2nd diffusion map and storing the 2nd diffusion map results into 'final_coordinates'"""
            )

        agg_coordinates = np.concatenate([val['coordinates'] for val in self.dm_dict.values()]).T
        final_dm_results = diffusion_mapping(agg_coordinates, **self.dm2_params) if len(self.dist_functions) > 1 else {'coordinates': agg_coordinates.T}
        self.results_dict['dm2'] = ndarray_to_df(final_dm_results['coordinates'].T, self.features, axis=0)

        self.best_features_idx, labels = self.return_best_features_by_kmeans(final_dm_results['coordinates'])
        self.best_features = self.features[self.best_features_idx]
        if self.verbose:
            logger.info(f'Best features by KMeans are: {self.best_features}')
            logger.info(f"Using KMeans algorithm in order to rank the features who are in final_coordinates")

    def transform(self, X):
        """
        Transforms X w/ all features in X with the best features only
        :param X: a scaled numpy matrix
        :return: return X with the best features only
        """
        self.X = X
        return self.X[self.best_features]

    def fit_transform(self, X, y):
        """
        Fits transformer to X and y
        :param X: a scaled numpy matrix
        :param y: an array of labels (int only)
        :return: a transformed X with the best features only
        """
        self.fit(X, y)
        return self.transform(X)
