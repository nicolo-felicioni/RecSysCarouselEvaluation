#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Massimo Quadrana
"""
import numpy as np

from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Recommender_utils import check_matrix
from Recommenders.DataIO import DataIO


class TopPop(BaseRecommender):
    """Top Popular recommender"""

    RECOMMENDER_NAME = "TopPopRecommender"

    def __init__(self, URM_train):
        super(TopPop, self).__init__(URM_train)


    def fit(self):

        # Use np.ediff1d and NOT a sum done over the rows as there might be values other than 0/1
        self.item_pop = np.ediff1d(self.URM_train.tocsc().indptr)
        self.n_items = self.URM_train.shape[1]


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        # Create a single (n_items, ) array with the item score, then copy it for every user

        if items_to_compute is not None:
            item_pop_to_copy = - np.ones(self.n_items, dtype=np.float32)*np.inf
            item_pop_to_copy[items_to_compute] = self.item_pop[items_to_compute].copy()
        else:
            item_pop_to_copy = self.item_pop.copy()

        item_scores = np.array(item_pop_to_copy, dtype=np.float32).reshape((1, -1))
        item_scores = np.repeat(item_scores, len(user_id_array), axis = 0)

        return item_scores


    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"item_pop": self.item_pop}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")


class TopPopFeature(BaseItemCBFRecommender):
    """Top Popular Feature recommender"""

    """
    This recommender takes the top popular items which have a certain feature specified.
    """

    RECOMMENDER_NAME = "TopPopFeatureRecommender"

    def __init__(self, URM_train, ICM_train, feature_name_to_idx_dict=None, verbose=True):
        super(TopPopFeature, self).__init__(URM_train, ICM_train, verbose=verbose)
        self.feature_name_to_idx_dict=feature_name_to_idx_dict
        self.ICM_train = self.ICM_train.tocsc()


    def fit(self, feature_name=None, feature_idx=None):
        """
        The fit takes in input the feature to take into consideration.
        It may be specified with its name (if in the constructor the dictionary was provided)
        or with its index
        :param feature_name: Optional, the name of the feature
        :param feature_idx: Optional, the index in the ICM of the feature
        :return: None
        """

        # assert that is possible to understand which is the index of the feature
        assert not(feature_name is None and feature_idx is None), "No feature name or index provided."
        assert not(feature_idx is None and self.feature_name_to_idx_dict is None), "Feature name provided, " \
                                                                                   "but the dictionary was " \
                                                                                   "not provided in the constructor"

        if feature_name is not None and feature_idx is not None:
            assert feature_idx == self.feature_name_to_idx_dict[feature_name], f"provided an inconsistent pair of " \
                                                                               f"index ({feature_idx}) " \
                                                                               f"and name ({feature_name})"

        # save the index in a class attribute
        self.feature_idx = feature_idx if feature_idx is not None else self.feature_name_to_idx_dict[feature_name]

        # Use np.ediff1d and NOT a sum done over the rows as there might be values other than 0/1
        self.item_pop = np.ediff1d(self.URM_train.tocsc().indptr)
        self.n_items = self.URM_train.shape[1]

        # the items to compute must be only the ones with the specified feature
        assert self.ICM_train.getformat() == 'csc'
        self.items_with_feature = self.ICM_train.indices[self.ICM_train.indptr[self.feature_idx]:
                                                    self.ICM_train.indptr[self.feature_idx+1]]


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        # Create a single (n_items, ) array with the item score, then copy it for every user

        # if items_to_compute provided, the intersection between items_to_compute and items_with_feature will be considered
        if items_to_compute is not None:
            items_to_compute = np.array(items_to_compute)
            items_to_compute = np.intersect1d(items_to_compute, self.items_with_feature)

        # else, the items_to_copmute are only the items_with_feature
        else:
            items_to_compute = self.items_with_feature

        item_pop_to_copy = - np.ones(self.n_items, dtype=np.float32) * np.inf
        item_pop_to_copy[items_to_compute] = self.item_pop[items_to_compute].copy()

        item_scores = np.array(item_pop_to_copy, dtype=np.float32).reshape((1, -1))
        item_scores = np.repeat(item_scores, len(user_id_array), axis = 0)

        return item_scores


    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"item_pop": self.item_pop,
                             "items_with_feature": self.items_with_feature,
                             "n_items": self.n_items}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")


class TopPopYearRange(BaseRecommender):
    """Top Popular Year Range recommender"""

    """
    This recommender takes the top popular items which have the 'year' attribute within some range specified.
    """

    RECOMMENDER_NAME = "TopPopYearRangeRecommender"

    def __init__(self, URM_train, ICM_train_year, verbose=False):
        super(TopPopYearRange, self).__init__(URM_train, verbose=verbose)

        # assuming that ICM_train_year is a dense feature composed of only one column
        self.ICM_train_year = np.array(ICM_train_year.todense(), dtype=int).reshape(-1)
        self.min_year = np.min(self.ICM_train_year)
        self.max_year = np.max(self.ICM_train_year)



    def fit(self, year_lower_bound=None, year_upper_bound=None):
        """
        The fit takes in input a range of years to take into consideration.
        The range will be considered with the 'range' function.
        ex.: range(year_lower_bound,year_upper_bound).
        If only one particular year is desired, then the bounds should be:
        year_lower_bound = year
        year_upper_bound = year+1

        :param year_lower_bound: Optional, the year lower bound
        :param year_upper_bound: Optional, the year upper bound
        :return: None
        """

        # assert that at least one bound is provided.
        assert not(year_lower_bound is None and year_upper_bound is None), "No year bound provided."

        if year_lower_bound is None:
            year_lower_bound = self.min_year

        if year_upper_bound is None:
            year_upper_bound = self.max_year + 1

        # Use np.ediff1d and NOT a sum done over the rows as there might be values other than 0/1
        self.item_pop = np.ediff1d(self.URM_train.tocsc().indptr)
        self.n_items = self.URM_train.shape[1]

        # save the array of the elements to consider in a class attribute
        self.items_in_range = np.array([item for item in range(self.n_items)
                                        if self.ICM_train_year[item] in range(year_lower_bound, year_upper_bound)])


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        # Create a single (n_items, ) array with the item score, then copy it for every user

        # if items_to_compute provided, the intersection between items_to_compute and  will be considered
        if items_to_compute is not None:
            items_to_compute = np.array(items_to_compute)
            items_to_compute = np.intersect1d(items_to_compute, self.items_in_range)

        # else, the items_to_copmute are only the items_with_feature
        else:
            items_to_compute = self.items_in_range

        item_pop_to_copy = - np.ones(self.n_items, dtype=np.float32) * np.inf
        item_pop_to_copy[items_to_compute] = self.item_pop[items_to_compute].copy()

        item_scores = np.array(item_pop_to_copy, dtype=np.float32).reshape((1, -1))
        item_scores = np.repeat(item_scores, len(user_id_array), axis=0)

        return item_scores


    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"item_pop": self.item_pop,
                             "items_in_range": self.items_in_range,
                             "n_items": self.n_items,
                             }

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")



class GlobalEffects(BaseRecommender):
    """docstring for GlobalEffects"""

    RECOMMENDER_NAME = "GlobalEffectsRecommender"

    def __init__(self, URM_train):
        super(GlobalEffects, self).__init__(URM_train)


    def fit(self, lambda_user=10, lambda_item=25):

        self.lambda_user = lambda_user
        self.lambda_item = lambda_item
        self.n_items = self.URM_train.shape[1]


        # convert to csc matrix for faster column-wise sum
        self.URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        # 1) global average
        self.mu = self.URM_train.data.sum(dtype=np.float32) / self.URM_train.data.shape[0]

        # 2) item average bias
        # compute the number of non-zero elements for each column
        col_nnz = np.diff(self.URM_train.indptr)

        # it is equivalent to:
        # col_nnz = X.indptr[1:] - X.indptr[:-1]
        # and it is **much faster** than
        # col_nnz = (X != 0).sum(axis=0)

        URM_train_unbiased = self.URM_train.copy()
        URM_train_unbiased.data -= self.mu
        self.item_bias = URM_train_unbiased.sum(axis=0) / (col_nnz + self.lambda_item)
        self.item_bias = np.asarray(self.item_bias).ravel()  # converts 2-d matrix to 1-d array without anycopy

        # 3) user average bias
        # NOTE: the user bias is *useless* for the sake of ranking items. We just show it here for educational purposes.

        # first subtract the item biases from each column
        # then repeat each element of the item bias vector a number of times equal to col_nnz
        # and subtract it from the data vector
        URM_train_unbiased.data -= np.repeat(self.item_bias, col_nnz)

        # now convert the csc matrix to csr for efficient row-wise computation
        URM_train_unbiased_csr = URM_train_unbiased.tocsr()
        row_nnz = np.diff(URM_train_unbiased_csr.indptr)
        # finally, let's compute the bias
        self.user_bias = URM_train_unbiased_csr.sum(axis=1).ravel() / (row_nnz + self.lambda_user)

        # 4) precompute the item ranking by using the item bias only
        # the global average and user bias won't change the ranking, so there is no need to use them
        #self.item_ranking = np.argsort(self.bi)[::-1]

        self.URM_train = check_matrix(self.URM_train, 'csr', dtype=np.float32)


    def _compute_item_score(self, user_id_array, items_to_compute=None):

        # Create a single (n_items, ) array with the item score, then copy it for every user

        if items_to_compute is not None:
            item_bias_to_copy = - np.ones(self.n_items, dtype=np.float32)*np.inf
            item_bias_to_copy[items_to_compute] = self.item_bias[items_to_compute].copy()
        else:
            item_bias_to_copy = self.item_bias.copy()

        item_scores = np.array(item_bias_to_copy, dtype=np.float).reshape((1, -1))
        item_scores = np.repeat(item_scores, len(user_id_array), axis = 0)

        return item_scores


    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"item_bias": self.item_bias}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")



class Random(BaseRecommender):
    """Random recommender"""

    RECOMMENDER_NAME = "RandomRecommender"

    def __init__(self, URM_train):
        super(Random, self).__init__(URM_train)


    def fit(self, random_seed=42):
        np.random.seed(random_seed)
        self.n_items = self.URM_train.shape[1]


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        # Create a random block (len(user_id_array), n_items) array with the item score

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf
            item_scores[:, items_to_compute] = np.random.rand(len(user_id_array), len(items_to_compute))

        else:
            item_scores = np.random.rand(len(user_id_array), self.n_items)

        return item_scores



    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")

