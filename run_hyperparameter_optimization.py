#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/06/2020

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import os, traceback, multiprocessing
from argparse import ArgumentParser
from functools import partial

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.NonPersonalizedRecommender import TopPopFeature, TopPopYearRange
from Recommenders.Recommender_import_list import *
from Data_manager import *

from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from HyperparameterTuning.run_parameter_search import runParameterSearch_Collaborative, runParameterSearch_Content, runParameterSearch_Hybrid

from Data_manager.data_consistency_check import assert_implicit_data, assert_disjoint_matrices

from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics
from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters
from Utils.all_dataset_stats_latex_table import all_dataset_stats_latex_table
from Data_manager.DataSplitter_Holdout import DataSplitter_Holdout
from Data_manager.DataPostprocessing_User_sample import DataPostprocessing_User_sample
from Data_manager.DataPostprocessing_K_Cores import DataPostprocessing_K_Cores



def read_data_split_and_search(dataset_class,
                               flag_baselines_tune=False,
                               flag_print_results=False):

    dataset_reader = dataset_class()

    if dataset_class is SpotifyChallenge2018Reader:
        dataset_reader = DataPostprocessing_User_sample(dataset_reader, user_quota = 0.1)
        dataset_reader = DataPostprocessing_K_Cores(dataset_reader, k_cores_value = 10)

    elif dataset_class is NetflixPrizeReader:
        dataset_reader = DataPostprocessing_User_sample(dataset_reader, user_quota = 0.2)


    result_folder_path = "result_experiments/{}/".format(dataset_reader._get_dataset_name())
    data_folder_path = result_folder_path + "data/"
    model_folder_path = result_folder_path + "models/"

    dataSplitter = DataSplitter_Holdout(dataset_reader, user_wise = False, split_interaction_quota_list=[80, 10, 10])
    dataSplitter.load_data(save_folder_path=data_folder_path)

    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
    URM_train_last_test = URM_train + URM_validation


    # Ensure IMPLICIT data and disjoint test-train split
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)


    plot_popularity_bias([URM_train + URM_validation, URM_test],
                         ["URM train", "URM test"],
                         data_folder_path + "item_popularity_plot")

    save_popularity_statistics([URM_train + URM_validation, URM_test],
                               ["URM train", "URM test"],
                               data_folder_path + "item_popularity_statistics")
    #
    # all_dataset_stats_latex_table(URM_train + URM_validation + URM_test, dataset_class._get_dataset_name(),
    #                               data_folder_path + "dataset_stats.tex")



    collaborative_algorithm_list = [
        Random,
        TopPop,
        GlobalEffects,
        UserKNNCFRecommender,
        ItemKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        PureSVDRecommender,
        NMFRecommender,
        IALSRecommender,
        MatrixFactorization_BPR_Cython,
        MatrixFactorization_FunkSVD_Cython,
        # MatrixFactorization_AsySVD_Cython,
        EASE_R_Recommender,
        SLIM_BPR_Cython,
        SLIMElasticNetRecommender,
        ]



    metric_to_optimize = 'MAP'
    cutoff_list_validation = [10]
    cutoff_list_test = [5, 10, 20]

    n_cases = 50
    n_random_starts = int(n_cases/3)

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list_validation)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list_test)


    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train=URM_train,
                                                       URM_train_last_test=URM_train_last_test,
                                                       metric_to_optimize=metric_to_optimize,
                                                       evaluator_validation_earlystopping=evaluator_validation,
                                                       evaluator_validation=evaluator_validation,
                                                       similarity_type_list = KNN_similarity_to_report_list,
                                                       evaluator_test=evaluator_test,
                                                       output_folder_path=model_folder_path,
                                                       resume_from_saved=True,
                                                       evaluate_on_test = "last",
                                                       parallelizeKNN=False,
                                                       allow_weighting=True,
                                                       n_cases=n_cases,
                                                       n_random_starts=n_random_starts)



    if flag_baselines_tune:

        pool = multiprocessing.Pool(processes=int(os.cpu_count()/3), maxtasksperchild=1)
        resultList = pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)

        pool.close()
        pool.join()

        # for recommender_class in collaborative_algorithm_list:
        #     try:
        #         runParameterSearch_Collaborative_partial(recommender_class)
        #     except Exception as e:
        #         print("On recommender {} Exception {}".format(recommender_class, str(e)))
        #         traceback.print_exc()


        ###############################################################################################
        ##### Item Content Baselines

        for ICM_name, ICM_object in dataSplitter.get_loaded_ICM_dict().items():

            try:

                if ICM_name == 'ICM_genres':
                    runParameterSearch_Content(TopPopFeature,
                                               URM_train=URM_train,
                                               URM_train_last_test=URM_train + URM_validation,
                                               metric_to_optimize=metric_to_optimize,
                                               evaluator_validation=evaluator_validation,
                                               similarity_type_list=KNN_similarity_to_report_list,
                                               evaluator_test=evaluator_test,
                                               output_folder_path=model_folder_path,
                                               parallelizeKNN=True,
                                               allow_weighting=True,
                                               resume_from_saved=True,
                                               evaluate_on_test="last",
                                               ICM_name=ICM_name,
                                               ICM_object=ICM_object.copy(),
                                               n_cases=n_cases,
                                               n_random_starts=n_random_starts)

                if ICM_name == 'ICM_year':
                    runParameterSearch_Content(TopPopYearRange,
                                               URM_train=URM_train,
                                               URM_train_last_test=URM_train + URM_validation,
                                               metric_to_optimize=metric_to_optimize,
                                               evaluator_validation=evaluator_validation,
                                               similarity_type_list=KNN_similarity_to_report_list,
                                               evaluator_test=evaluator_test,
                                               output_folder_path=model_folder_path,
                                               parallelizeKNN=True,
                                               allow_weighting=True,
                                               resume_from_saved=True,
                                               evaluate_on_test="last",
                                               ICM_name=ICM_name,
                                               ICM_object=ICM_object.copy(),
                                               n_cases=n_cases,
                                               n_random_starts=n_random_starts)

                runParameterSearch_Content(ItemKNNCBFRecommender,
                                            URM_train = URM_train,
                                            URM_train_last_test = URM_train + URM_validation,
                                            metric_to_optimize = metric_to_optimize,
                                            evaluator_validation = evaluator_validation,
                                            similarity_type_list = KNN_similarity_to_report_list,
                                            evaluator_test = evaluator_test,
                                            output_folder_path = model_folder_path,
                                            parallelizeKNN = True,
                                            allow_weighting = True,
                                            resume_from_saved = True,
                                            evaluate_on_test = "last",
                                            ICM_name = ICM_name,
                                            ICM_object = ICM_object.copy(),
                                            n_cases = n_cases,
                                            n_random_starts = n_random_starts)


                runParameterSearch_Hybrid(ItemKNN_CFCBF_Hybrid_Recommender,
                                            URM_train = URM_train,
                                            URM_train_last_test = URM_train + URM_validation,
                                            metric_to_optimize = metric_to_optimize,
                                            evaluator_validation = evaluator_validation,
                                            similarity_type_list = KNN_similarity_to_report_list,
                                            evaluator_test = evaluator_test,
                                            output_folder_path = model_folder_path,
                                            parallelizeKNN = True,
                                            allow_weighting = True,
                                            resume_from_saved = True,
                                            evaluate_on_test = "last",
                                            ICM_name = ICM_name,
                                            ICM_object = ICM_object.copy(),
                                            n_cases = n_cases,
                                            n_random_starts = n_random_starts)

            except Exception as e:

                print("On CBF recommender for ICM {} Exception {}".format(ICM_name, str(e)))
                traceback.print_exc()




        ################################################################################################
        ###### User Content Baselines

        for UCM_name, UCM_object in dataSplitter.get_loaded_UCM_dict().items():

            try:

                runParameterSearch_Content(UserKNNCBFRecommender,
                                            URM_train = URM_train,
                                            URM_train_last_test = URM_train + URM_validation,
                                            metric_to_optimize = metric_to_optimize,
                                            evaluator_validation = evaluator_validation,
                                            similarity_type_list = KNN_similarity_to_report_list,
                                            evaluator_test = evaluator_test,
                                            output_folder_path = model_folder_path,
                                            parallelizeKNN = True,
                                            allow_weighting = True,
                                            resume_from_saved = True,
                                            evaluate_on_test = "last",
                                            ICM_name = UCM_name,
                                            ICM_object = UCM_object.copy(),
                                            n_cases = n_cases,
                                            n_random_starts = n_random_starts)



                runParameterSearch_Hybrid(UserKNN_CFCBF_Hybrid_Recommender,
                                            URM_train = URM_train,
                                            URM_train_last_test = URM_train + URM_validation,
                                            metric_to_optimize = metric_to_optimize,
                                            evaluator_validation = evaluator_validation,
                                            similarity_type_list = KNN_similarity_to_report_list,
                                            evaluator_test = evaluator_test,
                                            output_folder_path = model_folder_path,
                                            parallelizeKNN = True,
                                            allow_weighting = True,
                                            resume_from_saved = True,
                                            evaluate_on_test = "last",
                                            ICM_name = UCM_name,
                                            ICM_object = UCM_object.copy(),
                                            n_cases = n_cases,
                                            n_random_starts = n_random_starts)

            except Exception as e:

                print("On CBF recommender for UCM {} Exception {}".format(UCM_name, str(e)))
                traceback.print_exc()





    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:

        n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)

        result_loader = ResultFolderLoader(model_folder_path,
                                           base_algorithm_list = None,
                                           other_algorithm_list = None,
                                           KNN_similarity_list = KNN_similarity_to_report_list,
                                           ICM_names_list = dataSplitter.get_loaded_ICM_dict().keys(),
                                           UCM_names_list = dataSplitter.get_loaded_UCM_dict().keys(),
                                           )

        result_loader.generate_latex_results(result_folder_path + "{}_latex_results.txt".format("accuracy_metrics"),
                                           metrics_list = ['RECALL', 'PRECISION', 'MAP', 'NDCG'],
                                           cutoffs_list = cutoff_list_test,
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_results(result_folder_path + "{}_latex_results.txt".format("beyond_accuracy_metrics"),
                                           metrics_list = ["NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "COVERAGE_ITEM",
                                                           "DIVERSITY_GINI", "SHANNON_ENTROPY", "COVERAGE_ITEM_CORRECT",
                                                           "COVERAGE_USER_CORRECT", "AVERAGE_POPULARITY"],
                                           cutoffs_list = cutoff_list_validation,
                                           table_title = None,
                                           highlight_best = True)

        result_loader.generate_latex_time_statistics(result_folder_path + "{}_latex_results.txt".format("time"),
                                           n_evaluation_users=n_test_users,
                                           table_title = None)



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help='Baseline hyperparameter search', type=bool, default=True)
    parser.add_argument('-p', '--print_results',        help='Print results', type=bool, default=True)

    input_flags = parser.parse_args()
    print(input_flags)

    KNN_similarity_to_report_list = ['cosine']#, 'dice', 'jaccard', 'asymmetric', 'tversky', 'euclidean']

    dataset_list = [Movielens10MReader, NetflixPrizeReader]#, SpotifyChallenge2018Reader]

    for dataset_class in dataset_list:
        read_data_split_and_search(dataset_class,
                                   flag_baselines_tune=input_flags.baseline_tune,
                                   flag_print_results=input_flags.print_results,
                                   )

