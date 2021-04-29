"""
Created on 02/10/2020

@author: anonymous for blind review
"""
from EvaluatorMultipleCarousels import EvaluatorMultipleCarousels
from Recommenders.NonPersonalizedRecommender import TopPopYearRange, TopPopFeature
from Recommenders.Recommender_import_list import *
from Data_manager import *

from Data_manager import *
from enum import Enum



from Evaluation.Evaluator import *
from Evaluation.Evaluator import _create_empty_metrics_dict, EvaluatorMetrics
import traceback

from Evaluation.metrics import dcg



from Data_manager.DataSplitter_Holdout import DataSplitter_Holdout
from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
from Recommenders.DataIO import DataIO

def run_carousel_eval(dataset_class, carousel_recommender_class_list):

    dataset_reader = dataset_class()

    result_folder_path = "result_experiments/{}/".format(dataset_reader._get_dataset_name())
    data_folder_path = result_folder_path + "data/"
    model_folder_path = result_folder_path + "models/"
    carousel_evaluation_folder_path = result_folder_path + "carousel_eval/"

    dataSplitter = DataSplitter_Holdout(dataset_reader, user_wise = False, split_interaction_quota_list=[80, 10, 10])
    dataSplitter.load_data(save_folder_path=data_folder_path)

    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
    URM_train_last_test = URM_train + URM_validation

    if dataset_reader._get_dataset_name() == 'Movielens1M':
        ICM_name = "ICM_genres"
        UCM_name = "UCM_all"
        ICM_dict = dataSplitter.get_loaded_ICM_dict()
        ICM_object = dataSplitter.get_loaded_ICM_dict()[ICM_name]
        ICM_year = dataSplitter.get_loaded_ICM_dict()['ICM_year']
        UCM_object = dataSplitter.get_loaded_UCM_dict()[UCM_name]
    elif dataset_reader._get_dataset_name() == 'Movielens10M':
        ICM_name = "ICM_genres"
        UCM_name = ""
        ICM_dict = dataSplitter.get_loaded_ICM_dict()
        ICM_object = dataSplitter.get_loaded_ICM_dict()[ICM_name]
        ICM_year = dataSplitter.get_loaded_ICM_dict()['ICM_year']
        UCM_object = None
    else:
        ICM_name = ""
        UCM_name = ""
        ICM_dict = dataSplitter.get_loaded_ICM_dict()
        ICM_object = None
        ICM_year = None
        UCM_object = None

    ################################################################################################
    ######
    ######      CREATE CAROUSEL EVALUATOR
    ######

    cutoff_list_test = [10]


    def _get_trained_carousel_recommenders(carousel_recommender_class_list, URM_train, ICM_dict=None):

        carousel_recommender_instance_list = []

        for recommender_class in carousel_recommender_class_list:


            if recommender_class == TopPopFeature:
                ICM_genres = ICM_dict['ICM_genres']
                recommender_object = recommender_class(URM_train, ICM_genres)
                file_name = recommender_object.RECOMMENDER_NAME + "_ICM_genres_best_model_last"

            elif recommender_class == TopPopYearRange:
                ICM_year = ICM_dict['ICM_year']
                recommender_object = recommender_class(URM_train, ICM_year)
                file_name = recommender_object.RECOMMENDER_NAME + "_ICM_year_best_model_last"

            elif recommender_class == ItemKNNCBFRecommender:
                # TODO HARDCODED ICM GENRES
                ICM_genres = ICM_dict['ICM_genres']
                recommender_object = recommender_class(URM_train, ICM_genres)
                # TODO HARDCODED ICM GENRES
                file_name = recommender_object.RECOMMENDER_NAME + "_ICM_genres_cosine_best_model_last"

            else:
                recommender_object = recommender_class(URM_train)

                if recommender_class in [ItemKNNCFRecommender, UserKNNCFRecommender]:
                    file_name = recommender_object.RECOMMENDER_NAME + "_cosine_best_model_last"
                else:
                    file_name = recommender_object.RECOMMENDER_NAME + "_best_model_last"


            recommender_object.load_model(model_folder_path, file_name=file_name)

            carousel_recommender_instance_list.append(recommender_object)

        return carousel_recommender_instance_list



    carousel_recommender_instance_list = _get_trained_carousel_recommenders(carousel_recommender_class_list, URM_train_last_test, ICM_dict=ICM_dict)

    evaluator_test = EvaluatorMultipleCarousels(URM_test, cutoff_list=cutoff_list_test, exclude_seen=True,
                                                carousel_recommender_list= carousel_recommender_instance_list)


    ################################################################################################
    ######
    ######      EVALUATE MODELS
    ######


    recommender_class_list = [
        # Random,
        TopPop,
        TopPopFeature,
        TopPopYearRange,
        GlobalEffects,
        UserKNNCFRecommender,
        ItemKNNCFRecommender,
        UserKNNCBFRecommender,
        ItemKNNCBFRecommender,
        ItemKNN_CFCBF_Hybrid_Recommender,
        UserKNN_CFCBF_Hybrid_Recommender,
        P3alphaRecommender,
        RP3betaRecommender,
        SLIM_BPR_Cython,
        SLIMElasticNetRecommender,
        MatrixFactorization_BPR_Cython,
        MatrixFactorization_FunkSVD_Cython,
        # MatrixFactorization_AsySVD_Cython,
        PureSVDRecommender,
        NMFRecommender,
        IALSRecommender,
        EASE_R_Recommender,
        ]


    def _get_instance(recommender_class, URM_train, ICM, UCM_all, ICM_year=None):

        if issubclass(recommender_class, BaseItemCBFRecommender):
            recommender_object = recommender_class(URM_train, ICM)
        elif issubclass(recommender_class, BaseUserCBFRecommender):
            recommender_object = recommender_class(URM_train, UCM_all)
        elif recommender_class == TopPopYearRange:
            recommender_object = recommender_class(URM_train, ICM_year)
        else:
            recommender_object = recommender_class(URM_train)

        return recommender_object




    dataIO = DataIO(carousel_evaluation_folder_path)


    for index, recommender_class in enumerate(recommender_class_list):

        try:

            print("Evaluating [{}/{}]".format(index+1, len(recommender_class_list)))

            recommender_instance = _get_instance(recommender_class, URM_train_last_test, ICM_object, UCM_object, ICM_year)

            if recommender_class in [ItemKNNCFRecommender, UserKNNCFRecommender]:
                file_name = recommender_instance.RECOMMENDER_NAME + "_{}".format("cosine")
            elif recommender_class in [ItemKNNCBFRecommender, ItemKNN_CFCBF_Hybrid_Recommender]:
                file_name = recommender_instance.RECOMMENDER_NAME + "_{}_{}".format(ICM_name, "cosine")
            elif recommender_class in [UserKNNCBFRecommender, UserKNN_CFCBF_Hybrid_Recommender]:
                file_name = recommender_instance.RECOMMENDER_NAME + "_{}_{}".format(UCM_name, "cosine")
            elif recommender_class == TopPopFeature:
                file_name = recommender_instance.RECOMMENDER_NAME + "_ICM_genres"
            elif recommender_class == TopPopYearRange:
                file_name = recommender_instance.RECOMMENDER_NAME + "_ICM_year"
            else:
                file_name = recommender_instance.RECOMMENDER_NAME

            recommender_instance.load_model(model_folder_path, file_name=file_name + "_best_model_last")
            result_dict, _ = evaluator_test.evaluateRecommender(recommender_instance)

            data_dict_to_save = {"result_on_last":result_dict}
            dataIO.save_data(file_name + "_metadata.zip", data_dict_to_save)

        except Exception as e:
            print("On Recommender {} Exception {}".format(recommender_class, str(e)))
            traceback.print_exc()

    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    from Utils.ResultFolderLoader import ResultFolderLoader

    result_loader = ResultFolderLoader(carousel_evaluation_folder_path,
                                       base_algorithm_list = None,
                                       other_algorithm_list = None,
                                       KNN_similarity_list = KNN_similarity_to_report_list,
                                       ICM_names_list = [ICM_name],
                                       UCM_names_list = [UCM_name])

    result_loader.generate_latex_results(carousel_evaluation_folder_path + "{}_latex_results.txt".format(f"carousel_"
                                                                                                         f"{carousel_recommender_instance_list[0].RECOMMENDER_NAME}"
                                                                                                         f"_accuracy_metrics"),
                                         metrics_list=['RECALL', 'PRECISION', 'MAP', 'NDCG', 'NDCG_2D'],
                                         cutoffs_list=cutoff_list_test,
                                         table_title=None,
                                         highlight_best=True)

    result_loader.generate_latex_results(carousel_evaluation_folder_path + "{}_latex_results.txt".format(f"carousel_"
                                                                                                         f"{carousel_recommender_instance_list[0].RECOMMENDER_NAME}"
                                                                                                         f"_beyond_accuracy_metrics"),
                                         metrics_list=["NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "COVERAGE_ITEM",
                                                       "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                                         cutoffs_list=cutoff_list_test,
                                         table_title=None,
                                         highlight_best=True)



if __name__ == '__main__':

    KNN_similarity_to_report_list = ['cosine', ] #'dice', 'jaccard', 'asymmetric', 'tversky', 'euclidean']

    dataset_list = [Movielens10MReader, NetflixPrizeReader]

    carousel_recommender_class_list = [
        SLIMElasticNetRecommender,
    ]

    for dataset_class in dataset_list:
        run_carousel_eval(dataset_class,
                          carousel_recommender_class_list= carousel_recommender_class_list,
                          )

