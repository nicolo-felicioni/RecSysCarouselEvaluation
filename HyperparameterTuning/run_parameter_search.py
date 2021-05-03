#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

import os, multiprocessing
from functools import partial


######################################################################
##########                                                  ##########
##########                  PURE COLLABORATIVE              ##########
##########                                                  ##########
######################################################################

# KNN
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

# KNN machine learning
from Recommenders.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

# Matrix Factorization
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender, PureSVDItemRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython,\
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython




######################################################################
##########                                                  ##########
##########                  PURE CONTENT BASED              ##########
##########                                                  ##########
######################################################################
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender

from Recommenders.NonPersonalizedRecommender import TopPop, Random, GlobalEffects, TopPopFeature, TopPopYearRange


######################################################################
##########                                                  ##########
##########                       HYBRID                     ##########
##########                                                  ##########
######################################################################
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.UserKNN_CFCBF_Hybrid_Recommender import UserKNN_CFCBF_Hybrid_Recommender

######################################################################
from skopt.space import Real, Integer, Categorical
import traceback

from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

######################################################################


def runParameterSearch_Hybrid(recommender_class, URM_train, ICM_object, ICM_name, URM_train_last_test = None,
                                n_cases = 30, n_random_starts = 5, resume_from_saved = False,
                                save_model = "best", evaluate_on_test = "best",
                                evaluator_validation= None, evaluator_test=None, metric_to_optimize = "PRECISION",
                                output_folder_path ="result_experiments/", parallelizeKNN = False, allow_weighting = True,
                                similarity_type_list = None):
    """
    This function performs the hyperparameter optimization for a hybrid collaborative and content-based recommender

    :param recommender_class:   Class of the recommender object to optimize, it must be a BaseRecommender type
    :param URM_train:           Sparse matrix containing the URM training data
    :param ICM_object:          Sparse matrix containing the ICM training data
    :param ICM_name:            String containing the name of the ICM, will be used for the name of the output files
    :param URM_train_last_test: Sparse matrix containing the union of URM training and validation data to be used in the last evaluation
    :param n_cases:             Number of hyperparameter sets to explore
    :param n_random_starts:     Number of the initial random hyperparameter values to explore, usually set at 30% of n_cases
    :param resume_from_saved:   Boolean value, if True the optimization is resumed from the saved files, if False a new one is done
    :param save_model:          ["no", "best", "last"] which of the models to save, see HyperparameterTuning/SearchAbstractClass for details
    :param evaluate_on_test:    ["all", "best", "last", "no"] when to evaluate the model on the test data, see HyperparameterTuning/SearchAbstractClass for details
    :param evaluator_validation:    Evaluator object to be used for the validation of each hyperparameter set
    :param evaluator_test:          Evaluator object to be used for the test results, the output will only be saved but not used
    :param metric_to_optimize:  String with the name of the metric to be optimized as contained in the output of the evaluator objects
    :param output_folder_path:  Folder in which to save the output files
    :param parallelizeKNN:      Boolean value, if True the various heuristics of the KNNs will be computed in parallel, if False sequentially
    :param allow_weighting:     Boolean value, if True it enables the use of TF-IDF and BM25 to weight features, users and items in KNNs
    :param similarity_type_list: List of strings with the similarity heuristics to be used for the KNNs
    """

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    URM_train = URM_train.copy()
    ICM_object = ICM_object.copy()

    if URM_train_last_test is not None:
        URM_train_last_test = URM_train_last_test.copy()


   ##########################################################################################################

    output_file_name_root = recommender_class.RECOMMENDER_NAME + "_{}".format(ICM_name)

    parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)


    if recommender_class in [ItemKNN_CFCBF_Hybrid_Recommender, UserKNN_CFCBF_Hybrid_Recommender]:

        if similarity_type_list is None:
            similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]


        hyperparameters_range_dictionary = {}

        if recommender_class is ItemKNN_CFCBF_Hybrid_Recommender:
            hyperparameters_range_dictionary["ICM_weight"] = Real(low = 1e-2, high = 1e2, prior = 'log-uniform')

        elif recommender_class is UserKNN_CFCBF_Hybrid_Recommender:
            hyperparameters_range_dictionary["UCM_weight"] = Real(low = 1e-2, high = 1e2, prior = 'log-uniform')




        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_object],
            CONSTRUCTOR_KEYWORD_ARGS = {},
            FIT_POSITIONAL_ARGS = [],
            FIT_KEYWORD_ARGS = {}
        )


        if URM_train_last_test is not None:
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
        else:
            recommender_input_args_last_test = None


        run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNRecommender_on_similarity_type,
                                                       parameter_search_space = hyperparameters_range_dictionary,
                                                       recommender_input_args = recommender_input_args,
                                                       parameterSearch = parameterSearch,
                                                       resume_from_saved = resume_from_saved,
                                                       save_model = save_model,
                                                       evaluate_on_test = evaluate_on_test,
                                                       n_cases = n_cases,
                                                       n_random_starts = n_random_starts,
                                                       output_folder_path = output_folder_path,
                                                       output_file_name_root = output_file_name_root,
                                                       metric_to_optimize = metric_to_optimize,
                                                       allow_weighting = allow_weighting,
                                                       recommender_input_args_last_test = recommender_input_args_last_test)



        if parallelizeKNN:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
            resultList = pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

            pool.close()
            pool.join()

        else:

            for similarity_type in similarity_type_list:
                run_KNNCFRecommender_on_similarity_type_partial(similarity_type)


        return












def run_KNNRecommender_on_similarity_type(similarity_type, parameterSearch,
                                          parameter_search_space,
                                          recommender_input_args,
                                          n_cases,
                                          n_random_starts,
                                          resume_from_saved,
                                          save_model,
                                          evaluate_on_test,
                                          output_folder_path,
                                          output_file_name_root,
                                          metric_to_optimize,
                                          allow_weighting = False,
                                          allow_bias_ICM = False,
                                          allow_bias_URM = False,
                                          recommender_input_args_last_test = None):

    original_parameter_search_space = parameter_search_space

    hyperparameters_range_dictionary = {}
    hyperparameters_range_dictionary["topK"] = Integer(5, 1000)
    hyperparameters_range_dictionary["shrink"] = Integer(0, 1000)
    hyperparameters_range_dictionary["similarity"] = Categorical([similarity_type])
    hyperparameters_range_dictionary["normalize"] = Categorical([True, False])

    is_set_similarity = similarity_type in ["tversky", "dice", "jaccard", "tanimoto"]

    if similarity_type == "asymmetric":
        hyperparameters_range_dictionary["asymmetric_alpha"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparameters_range_dictionary["normalize"] = Categorical([True])

    elif similarity_type == "tversky":
        hyperparameters_range_dictionary["tversky_alpha"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparameters_range_dictionary["tversky_beta"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparameters_range_dictionary["normalize"] = Categorical([True])

    elif similarity_type == "euclidean":
        hyperparameters_range_dictionary["normalize"] = Categorical([True, False])
        hyperparameters_range_dictionary["normalize_avg_row"] = Categorical([True, False])
        hyperparameters_range_dictionary["similarity_from_distance_mode"] = Categorical(["lin", "log", "exp"])


    if not is_set_similarity:

        if allow_weighting:
            hyperparameters_range_dictionary["feature_weighting"] = Categorical(["none", "BM25", "TF-IDF"])

        if allow_bias_ICM:
            hyperparameters_range_dictionary["ICM_bias"] = Real(low = 1e-2, high = 1e+3, prior = 'log-uniform')

        if allow_bias_URM:
            hyperparameters_range_dictionary["URM_bias"] = Real(low = 1e-2, high = 1e+3, prior = 'log-uniform')

    local_parameter_search_space = {**hyperparameters_range_dictionary, **original_parameter_search_space}

    parameterSearch.search(recommender_input_args,
                           parameter_search_space = local_parameter_search_space,
                           n_cases = n_cases,
                           n_random_starts = n_random_starts,
                           resume_from_saved = resume_from_saved,
                           save_model = save_model,
                           evaluate_on_test = evaluate_on_test,
                           output_folder_path = output_folder_path,
                           output_file_name_root = output_file_name_root + "_" + similarity_type,
                           metric_to_optimize = metric_to_optimize,
                           recommender_input_args_last_test = recommender_input_args_last_test)





def runParameterSearch_Content(recommender_class, URM_train, ICM_object, ICM_name, URM_train_last_test = None,
                                n_cases = 30, n_random_starts = 5, resume_from_saved = False,
                                save_model = "best", evaluate_on_test = "best",
                                evaluator_validation= None, evaluator_test=None, metric_to_optimize = "PRECISION",
                                output_folder_path ="result_experiments/", parallelizeKNN = False, allow_weighting = True, allow_bias_ICM = False,
                                similarity_type_list = None):
    """
    This function performs the hyperparameter optimization for a content-based recommender

    :param recommender_class:   Class of the recommender object to optimize, it must be a BaseRecommender type
    :param URM_train:           Sparse matrix containing the URM training data
    :param ICM_object:          Sparse matrix containing the ICM training data
    :param ICM_name:            String containing the name of the ICM, will be used for the name of the output files
    :param URM_train_last_test: Sparse matrix containing the union of URM training and validation data to be used in the last evaluation
    :param n_cases:             Number of hyperparameter sets to explore
    :param n_random_starts:     Number of the initial random hyperparameter values to explore, usually set at 30% of n_cases
    :param resume_from_saved:   Boolean value, if True the optimization is resumed from the saved files, if False a new one is done
    :param save_model:          ["no", "best", "last"] which of the models to save, see HyperparameterTuning/SearchAbstractClass for details
    :param evaluate_on_test:    ["all", "best", "last", "no"] when to evaluate the model on the test data, see HyperparameterTuning/SearchAbstractClass for details
    :param evaluator_validation:    Evaluator object to be used for the validation of each hyperparameter set
    :param evaluator_test:          Evaluator object to be used for the test results, the output will only be saved but not used
    :param metric_to_optimize:  String with the name of the metric to be optimized as contained in the output of the evaluator objects
    :param output_folder_path:  Folder in which to save the output files
    :param parallelizeKNN:      Boolean value, if True the various heuristics of the KNNs will be computed in parallel, if False sequentially
    :param allow_weighting:     Boolean value, if True it enables the use of TF-IDF and BM25 to weight features, users and items in KNNs
    :param allow_bias_ICM:      Boolean value, if True it enables the use of bias to shift the values of the ICM
    :param similarity_type_list: List of strings with the similarity heuristics to be used for the KNNs
    """


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    URM_train = URM_train.copy()
    ICM_object = ICM_object.copy()

    if URM_train_last_test is not None:
        URM_train_last_test = URM_train_last_test.copy()


    assert recommender_class in [ItemKNNCBFRecommender, UserKNNCBFRecommender, TopPopFeature, TopPopYearRange]

   ##########################################################################################################

    output_file_name_root = recommender_class.RECOMMENDER_NAME + "_{}".format(ICM_name)

    if recommender_class in [TopPopFeature, ]:
        """
        TopPopFeature (hardcoded for Movielens10M)
        """

        parameterSearch = SearchSingleCase(recommender_class, evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test)

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, ICM_object],
            CONSTRUCTOR_KEYWORD_ARGS={},
            FIT_POSITIONAL_ARGS=[],
            FIT_KEYWORD_ARGS={}
        )

        if URM_train_last_test is not None:
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
        else:
            recommender_input_args_last_test = None

        parameterSearch.search(recommender_input_args,
                               recommender_input_args_last_test=recommender_input_args_last_test,
                               fit_hyperparameters_values={'feature_idx': 6}, # the most popular genre in Movielens 10M
                               output_folder_path=output_folder_path,
                               output_file_name_root=output_file_name_root,
                               resume_from_saved=resume_from_saved,
                               save_model=save_model,
                               evaluate_on_test=evaluate_on_test,
                               )

        return

    if recommender_class in [TopPopYearRange]:
        """
        TopPopYearRange (hardcoded for last two years)
        """

        parameterSearch = SearchSingleCase(recommender_class, evaluator_validation=evaluator_validation,
                                           evaluator_test=evaluator_test)

        recommender_input_args = SearchInputRecommenderArgs(
            CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, ICM_object],
            CONSTRUCTOR_KEYWORD_ARGS={},
            FIT_POSITIONAL_ARGS=[],
            FIT_KEYWORD_ARGS={}
        )

        if URM_train_last_test is not None:
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
        else:
            recommender_input_args_last_test = None

        parameterSearch.search(recommender_input_args,
                               recommender_input_args_last_test=recommender_input_args_last_test,
                               fit_hyperparameters_values={'year_lower_bound': int(max(ICM_object.data))-2},  # last two years
                               output_folder_path=output_folder_path,
                               output_file_name_root=output_file_name_root,
                               resume_from_saved=resume_from_saved,
                               save_model=save_model,
                               evaluate_on_test=evaluate_on_test,
                               )

        return


    # IF IT IS NOT A NON-PERSONALIZED:

    parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)


    if similarity_type_list is None:
        similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]


    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, ICM_object],
        CONSTRUCTOR_KEYWORD_ARGS = {},
        FIT_POSITIONAL_ARGS = [],
        FIT_KEYWORD_ARGS = {}
    )



    if URM_train_last_test is not None:
        recommender_input_args_last_test = recommender_input_args.copy()
        recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
    else:
        recommender_input_args_last_test = None


    run_KNNCBFRecommender_on_similarity_type_partial = partial(run_KNNRecommender_on_similarity_type,
                                                   recommender_input_args = recommender_input_args,
                                                   parameter_search_space = {},
                                                   parameterSearch = parameterSearch,
                                                   n_cases = n_cases,
                                                   n_random_starts = n_random_starts,
                                                   resume_from_saved = resume_from_saved,
                                                   save_model = save_model,
                                                   evaluate_on_test = evaluate_on_test,
                                                   output_folder_path = output_folder_path,
                                                   output_file_name_root = output_file_name_root,
                                                   metric_to_optimize = metric_to_optimize,
                                                   allow_weighting = allow_weighting,
                                                   allow_bias_ICM = allow_bias_ICM,
                                                   recommender_input_args_last_test = recommender_input_args_last_test)



    if parallelizeKNN:
        pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
        pool.map(run_KNNCBFRecommender_on_similarity_type_partial, similarity_type_list)

        pool.close()
        pool.join()

    else:

        for similarity_type in similarity_type_list:
            run_KNNCBFRecommender_on_similarity_type_partial(similarity_type)








def runParameterSearch_Collaborative(recommender_class, URM_train, URM_train_last_test = None,
                                     n_cases = 35, n_random_starts = 5, resume_from_saved = False,
                                     save_model = "best",  evaluate_on_test = "best",
                                     evaluator_validation = None, evaluator_test = None, evaluator_validation_earlystopping = None,
                                     metric_to_optimize = "PRECISION",
                                     output_folder_path ="result_experiments/", parallelizeKNN = True,
                                     allow_weighting = True, allow_bias_URM=False, allow_dropout_MF = False, similarity_type_list = None):
    """
    This function performs the hyperparameter optimization for a collaborative recommender

    :param recommender_class:   Class of the recommender object to optimize, it must be a BaseRecommender type
    :param URM_train:           Sparse matrix containing the URM training data
    :param URM_train_last_test: Sparse matrix containing the union of URM training and validation data to be used in the last evaluation
    :param n_cases:             Number of hyperparameter sets to explore
    :param n_random_starts:     Number of the initial random hyperparameter values to explore, usually set at 30% of n_cases
    :param resume_from_saved:   Boolean value, if True the optimization is resumed from the saved files, if False a new one is done
    :param save_model:          ["no", "best", "last"] which of the models to save, see HyperparameterTuning/SearchAbstractClass for details
    :param evaluate_on_test:    ["all", "best", "last", "no"] when to evaluate the model on the test data, see HyperparameterTuning/SearchAbstractClass for details
    :param evaluator_validation:    Evaluator object to be used for the validation of each hyperparameter set
    :param evaluator_validation_earlystopping:   Evaluator object to be used for the earlystopping of ML algorithms, can be the same of evaluator_validation
    :param evaluator_test:          Evaluator object to be used for the test results, the output will only be saved but not used
    :param metric_to_optimize:  String with the name of the metric to be optimized as contained in the output of the evaluator objects
    :param output_folder_path:  Folder in which to save the output files
    :param parallelizeKNN:      Boolean value, if True the various heuristics of the KNNs will be computed in parallel, if False sequentially
    :param allow_weighting:     Boolean value, if True it enables the use of TF-IDF and BM25 to weight features, users and items in KNNs
    :param allow_bias_URM:      Boolean value, if True it enables the use of bias to shift the values of the URM
    :param allow_dropout_MF:    Boolean value, if True it enables the use of dropout on the latent factors of MF algorithms
    :param similarity_type_list: List of strings with the similarity heuristics to be used for the KNNs
    """



    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    earlystopping_keywargs = {"validation_every_n": 5,
                              "stop_on_validation": True,
                              "evaluator_object": evaluator_validation_earlystopping,
                              "lower_validations_allowed": 5,
                              "validation_metric": metric_to_optimize,
                              }

    URM_train = URM_train.copy()

    if URM_train_last_test is not None:
        URM_train_last_test = URM_train_last_test.copy()

    try:

        output_file_name_root = recommender_class.RECOMMENDER_NAME

        parameterSearch = SearchBayesianSkopt(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)




        if recommender_class in [TopPop, GlobalEffects, Random]:
            """
            TopPop, GlobalEffects and Random have no parameters therefore only one evaluation is needed
            """


            parameterSearch = SearchSingleCase(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )


            if URM_train_last_test is not None:
                recommender_input_args_last_test = recommender_input_args.copy()
                recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
            else:
                recommender_input_args_last_test = None


            parameterSearch.search(recommender_input_args,
                                   recommender_input_args_last_test = recommender_input_args_last_test,
                                   fit_hyperparameters_values={},
                                   output_folder_path = output_folder_path,
                                   output_file_name_root = output_file_name_root,
                                   resume_from_saved = resume_from_saved,
                                   save_model = save_model,
                                   evaluate_on_test = evaluate_on_test,
                                   )


            return



        ##########################################################################################################

        if recommender_class in [ItemKNNCFRecommender, UserKNNCFRecommender]:

            if similarity_type_list is None:
                similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )


            if URM_train_last_test is not None:
                recommender_input_args_last_test = recommender_input_args.copy()
                recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
            else:
                recommender_input_args_last_test = None


            run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNRecommender_on_similarity_type,
                                                           recommender_input_args = recommender_input_args,
                                                           parameter_search_space = {},
                                                           parameterSearch = parameterSearch,
                                                           n_cases = n_cases,
                                                           n_random_starts = n_random_starts,
                                                           resume_from_saved = resume_from_saved,
                                                           save_model = save_model,
                                                           evaluate_on_test = evaluate_on_test,
                                                           output_folder_path = output_folder_path,
                                                           output_file_name_root = output_file_name_root,
                                                           metric_to_optimize = metric_to_optimize,
                                                           allow_weighting = allow_weighting,
                                                           allow_bias_URM = allow_bias_URM,
                                                           recommender_input_args_last_test = recommender_input_args_last_test)



            if parallelizeKNN:
                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
                pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

                pool.close()
                pool.join()

            else:

                for similarity_type in similarity_type_list:
                    run_KNNCFRecommender_on_similarity_type_partial(similarity_type)


            return



       ##########################################################################################################

        if recommender_class is P3alphaRecommender:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK"] = Integer(5, 1000)
            hyperparameters_range_dictionary["alpha"] = Real(low = 0, high = 2, prior = 'uniform')
            hyperparameters_range_dictionary["normalize_similarity"] = Categorical([True, False])

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )


        ##########################################################################################################

        if recommender_class is RP3betaRecommender:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK"] = Integer(5, 1000)
            hyperparameters_range_dictionary["alpha"] = Real(low = 0, high = 2, prior = 'uniform')
            hyperparameters_range_dictionary["beta"] = Real(low = 0, high = 2, prior = 'uniform')
            hyperparameters_range_dictionary["normalize_similarity"] = Categorical([True, False])

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )



        ##########################################################################################################

        if recommender_class is MatrixFactorization_FunkSVD_Cython:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["sgd_mode"] = Categorical(["sgd", "adagrad", "adam"])
            hyperparameters_range_dictionary["epochs"] = Categorical([500])
            hyperparameters_range_dictionary["use_bias"] = Categorical([True, False])
            hyperparameters_range_dictionary["batch_size"] = Categorical([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
            hyperparameters_range_dictionary["num_factors"] = Integer(1, 200)
            hyperparameters_range_dictionary["item_reg"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
            hyperparameters_range_dictionary["user_reg"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
            hyperparameters_range_dictionary["learning_rate"] = Real(low = 1e-4, high = 1e-1, prior = 'log-uniform')
            hyperparameters_range_dictionary["negative_interactions_quota"] = Real(low = 0.0, high = 0.5, prior = 'uniform')

            if allow_dropout_MF:
                hyperparameters_range_dictionary["dropout_quota"] = Real(low = 0.01, high = 0.7, prior = 'uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = earlystopping_keywargs
            )

        ##########################################################################################################

        if recommender_class is MatrixFactorization_AsySVD_Cython:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["sgd_mode"] = Categorical(["sgd", "adagrad", "adam"])
            hyperparameters_range_dictionary["epochs"] = Categorical([500])
            hyperparameters_range_dictionary["use_bias"] = Categorical([True, False])
            hyperparameters_range_dictionary["batch_size"] = Categorical([1])
            hyperparameters_range_dictionary["num_factors"] = Integer(1, 200)
            hyperparameters_range_dictionary["item_reg"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
            hyperparameters_range_dictionary["user_reg"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
            hyperparameters_range_dictionary["learning_rate"] = Real(low = 1e-4, high = 1e-1, prior = 'log-uniform')
            hyperparameters_range_dictionary["negative_interactions_quota"] = Real(low = 0.0, high = 0.5, prior = 'uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = earlystopping_keywargs
            )

        ##########################################################################################################

        if recommender_class is MatrixFactorization_BPR_Cython:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["sgd_mode"] = Categorical(["sgd", "adagrad", "adam"])
            hyperparameters_range_dictionary["epochs"] = Categorical([1500])
            hyperparameters_range_dictionary["num_factors"] = Integer(1, 200)
            hyperparameters_range_dictionary["batch_size"] = Categorical([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
            hyperparameters_range_dictionary["positive_reg"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
            hyperparameters_range_dictionary["negative_reg"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
            hyperparameters_range_dictionary["learning_rate"] = Real(low = 1e-4, high = 1e-1, prior = 'log-uniform')

            if allow_dropout_MF:
                hyperparameters_range_dictionary["dropout_quota"] = Real(low = 0.01, high = 0.7, prior = 'uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {**earlystopping_keywargs,
                                    "positive_threshold_BPR": None}
            )

        ##########################################################################################################

        if recommender_class is IALSRecommender:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["num_factors"] = Integer(1, 200)
            hyperparameters_range_dictionary["confidence_scaling"] = Categorical(["linear", "log"])
            hyperparameters_range_dictionary["alpha"] = Real(low = 1e-3, high = 50.0, prior = 'log-uniform')
            hyperparameters_range_dictionary["epsilon"] = Real(low = 1e-3, high = 10.0, prior = 'log-uniform')
            hyperparameters_range_dictionary["reg"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = earlystopping_keywargs
            )


        ##########################################################################################################

        if recommender_class is PureSVDRecommender:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["num_factors"] = Integer(1, 350)

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )


        ##########################################################################################################

        if recommender_class is PureSVDItemRecommender:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["num_factors"] = Integer(1, 350)
            hyperparameters_range_dictionary["topK"] = Integer(5, 1000)

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )


        ##########################################################################################################

        if recommender_class is NMFRecommender:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["num_factors"] = Integer(1, 350)
            hyperparameters_range_dictionary["solver"] = Categorical(["coordinate_descent", "multiplicative_update"])
            hyperparameters_range_dictionary["init_type"] = Categorical(["random", "nndsvda"])
            hyperparameters_range_dictionary["beta_loss"] = Categorical(["frobenius", "kullback-leibler"])

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )


        #########################################################################################################

        if recommender_class is SLIM_BPR_Cython:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK"] = Integer(5, 1000)
            hyperparameters_range_dictionary["epochs"] = Categorical([1500])
            hyperparameters_range_dictionary["symmetric"] = Categorical([True, False])
            hyperparameters_range_dictionary["sgd_mode"] = Categorical(["sgd", "adagrad", "adam"])
            hyperparameters_range_dictionary["lambda_i"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
            hyperparameters_range_dictionary["lambda_j"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')
            hyperparameters_range_dictionary["learning_rate"] = Real(low = 1e-4, high = 1e-1, prior = 'log-uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {**earlystopping_keywargs,
                                    "positive_threshold_BPR": None,
                                    'train_with_sparse_weights': None}
            )



        ##########################################################################################################

        if recommender_class is SLIMElasticNetRecommender:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK"] = Integer(5, 1000)
            hyperparameters_range_dictionary["l1_ratio"] = Real(low = 1e-5, high = 1.0, prior = 'log-uniform')
            hyperparameters_range_dictionary["alpha"] = Real(low = 1e-3, high = 1.0, prior = 'uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )


        #########################################################################################################

        if recommender_class is EASE_R_Recommender:

            hyperparameters_range_dictionary = {}
            hyperparameters_range_dictionary["topK"] = Categorical([None])#Integer(5, 3000)
            hyperparameters_range_dictionary["normalize_matrix"] = Categorical([False])
            hyperparameters_range_dictionary["l2_norm"] = Real(low = 1e0, high = 1e7, prior = 'log-uniform')

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                CONSTRUCTOR_KEYWORD_ARGS = {},
                FIT_POSITIONAL_ARGS = [],
                FIT_KEYWORD_ARGS = {}
            )


       #########################################################################################################

        if URM_train_last_test is not None:
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
        else:
            recommender_input_args_last_test = None

        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        parameterSearch.search(recommender_input_args,
                               parameter_search_space = hyperparameters_range_dictionary,
                               n_cases = n_cases,
                               n_random_starts = n_random_starts,
                               resume_from_saved = resume_from_saved,
                               save_model = save_model,
                               evaluate_on_test = evaluate_on_test,
                               output_folder_path = output_folder_path,
                               output_file_name_root = output_file_name_root,
                               metric_to_optimize = metric_to_optimize,
                               recommender_input_args_last_test = recommender_input_args_last_test)




    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()












def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """

    from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
    from Data_manager.DataSplitter_k_fold_stratified import DataSplitter_Warm_k_fold


    dataset_object = Movielens1MReader()

    dataSplitter = DataSplitter_Warm_k_fold(dataset_object)

    dataSplitter.load_data()

    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()


    output_folder_path = "result_experiments/SKOPT_prova/"


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)







    collaborative_algorithm_list = [
        Random,
        TopPop,
        P3alphaRecommender,
        RP3betaRecommender,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # PureSVDRecommender,
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender
    ]



    from Evaluation.Evaluator import EvaluatorHoldout

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[5])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[5, 10])


    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train = URM_train,
                                                       metric_to_optimize = "MAP",
                                                       n_cases = 8,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = output_folder_path)

    # pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    # resultList = pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)
    # pool.close()
    # pool.join()



    for recommender_class in collaborative_algorithm_list:

        try:

            runParameterSearch_Collaborative_partial(recommender_class)

        except Exception as e:

            print("On recommender {} Exception {}".format(recommender_class, str(e)))
            traceback.print_exc()








if __name__ == '__main__':


    read_data_split_and_search()
