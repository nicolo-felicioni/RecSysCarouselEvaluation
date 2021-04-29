from Evaluation.Evaluator import *
from Evaluation.Evaluator import _create_empty_metrics_dict, EvaluatorMetrics


class CarouselEvaluatorMetrics(Enum):
    NDCG_2D = "NDCG_2D"


def _create_empty_metrics_dict_carousel(cutoff_list, n_items, n_users, URM_train, URM_test, ignore_items,
                                        ignore_users, diversity_similarity_object):

    results_dict = _create_empty_metrics_dict(cutoff_list,
                                              n_items, n_users,
                                              URM_train,
                                              URM_test,
                                              ignore_items,
                                              ignore_users,
                                              diversity_similarity_object)

    for cutoff in cutoff_list:

        cutoff_dict = results_dict[cutoff]

        for metric in CarouselEvaluatorMetrics:
            cutoff_dict[metric.value] = 0.0

        results_dict[cutoff] = cutoff_dict

    return results_dict



# NDCG 2D

def dcg_2d_one_row(scores_i: np.ndarray, i: int):
    """
    Calculates the DCG_2D of the i-th row of the scores matrix.

    :param scores_i: the i-th row of the scores matrix
    :param i: the index of the given row
    :return: DCG_2D value for the given row
    """

    numerator_arr = np.power(2, scores_i) - 1
    denominator_arr = np.log2(np.arange(scores_i.shape[0], dtype=np.float32) + 2 + i)

    division_arr = np.divide(numerator_arr, denominator_arr)

    res = np.sum(division_arr, dtype=np.float32)

    return res


def dcg_2d(scores_mat: np.ndarray):

    n_rows = scores_mat.shape[0]

    dcg_2d_ = sum([dcg_2d_one_row(scores_mat[i], i) for i in range(n_rows)])

    return dcg_2d_


def idcg_2d(relevance: np.ndarray, original_n_rows: int, single_list_len: int, verbose=False):

    """
    Calculates the ideal DCG starting from the array of relevance and knowing how many items can be put on a single list
    :param relevance: relevance array
    :param single_list_len: length of a single row
    :return: idcg value
    """

    n_rows = original_n_rows
    n_cols = single_list_len

    # sort the relevance array
    relevance = np.sort(relevance)[::-1]

    if verbose:
        print(f"IDCG: creating a matrix full of zeros of {n_rows} rows and {n_cols} cols...")

    ideal_matrix = from_1d_to_mat_diag_relevance(start_1d_arr=relevance, n_rows=n_rows, n_cols=n_cols)

    return dcg_2d(ideal_matrix)


def from_1d_to_mat_diag_relevance(start_1d_arr, n_rows, n_cols):

    final_mat = np.empty(shape=(n_rows, n_cols))

    k = 0
    # diagonal assignment
    for sum_value in range(n_rows + n_cols - 1):
        for i in range(n_rows):
            j = sum_value - i
            if (i >= 0 and i < n_rows) and (j>=0 and j < n_cols):
                if k < len(start_1d_arr):
                    final_mat[i, j] = start_1d_arr[k]
                    k += 1
    return final_mat


def from_mat_diag_relevance_to_1d(start_mat):
    n_rows, n_cols = start_mat.shape

    final_len = n_rows*n_cols
    final_arr = np.empty(final_len)

    k = 0
    # diagonal assignment
    for sum_value in range(n_rows + n_cols - 1):
        for i in range(n_rows):
            j = sum_value - i
            if (i >= 0 and i < n_rows) and (j >= 0 and j < n_cols):
                if k < final_len:
                    final_arr[k] = start_mat[i, j]
                    k += 1

    return final_arr


def remove_duplicates_diag_relevance(ranked_list_2d, n_rows, n_cols):

    # from 1D to 2D
    mat = np.array(ranked_list_2d).reshape((n_rows, n_cols))

    # 1D array ordered by the diagonal heuristic
    ranked_list_by_rel = from_mat_diag_relevance_to_1d(mat)

    # remove duplicates
    _, unique_indices = np.unique(ranked_list_by_rel, return_index=True)

    duplicate_indices = np.invert(np.in1d(np.arange(len(ranked_list_by_rel)), unique_indices))

    ranked_list_by_rel_no_duplicates = np.copy(ranked_list_by_rel)
    ranked_list_by_rel_no_duplicates[duplicate_indices] = None

    mat_no_duplicates = from_1d_to_mat_diag_relevance(ranked_list_by_rel_no_duplicates, n_rows, n_cols)

    return mat_no_duplicates


def ndcg_2d(ranked_list_2d: list, pos_items: np.ndarray, single_list_len: int, relevance=None, verbose=False):

    """
    Calculates the NDCG_2D for the given ranked list of

    :param ranked_list_2d: 2D list of item ids ranked, this is the list to evaluate
    :param pos_items: array of the relevant item ids (ground truth)
    :param single_list_len: length of a single row of the 2D ranked_list
    :param relevance: list of the relevance of the corresponding item ids in pos_items.
                      If not provided, it is assumed to be a list of ones
    :return: NDCG 2D
    """

    # number of rows
    # integer (floor) division with //
    n_rows = len(ranked_list_2d)//single_list_len

    # assert that the len of the list to evaluate is a multiple of n_rows
    assert n_rows*single_list_len == len(ranked_list_2d), f"The list to evaluate is not a concatenation of lists" \
                                                          f" of length {single_list_len}. " \
                                                          f"The length of the list to evaluate is {len(ranked_list_2d)}"

    if verbose:
        print(f"n. of rows: {n_rows}")

    if relevance is None:
        relevance = np.ones_like(pos_items)
    assert len(relevance) == pos_items.shape[0]

    # Create a dictionary associating item_id to its relevance
    # it2rel[item] -> relevance[item]
    it2rel = {it: r for it, r in zip(pos_items, relevance)}

    mat_no_duplicates = remove_duplicates_diag_relevance(ranked_list_2d, n_rows=n_rows, n_cols=single_list_len)

    vectorized_get_rel = np.vectorize(lambda x: it2rel.get(x, 0.0))

    mat_relevance = vectorized_get_rel(mat_no_duplicates)

    rank_dcg_2d = dcg_2d(mat_relevance)

    ideal_dcg_2d = idcg_2d(relevance=relevance, original_n_rows=n_rows,
                           single_list_len=single_list_len, verbose=verbose)

    if rank_dcg_2d == 0.0:
        return 0.0

    ndcg_2d_ = rank_dcg_2d / ideal_dcg_2d

    return ndcg_2d_


class EvaluatorMultipleCarousels(Evaluator):
    """EvaluatorMultipleCarousels"""

    EVALUATOR_NAME = "EvaluatorMultipleCarousels"

    def __init__(self, URM_test_list, cutoff_list, min_ratings_per_user=1, exclude_seen=True,
                 diversity_object = None,
                 ignore_items = None,
                 ignore_users = None,
                 verbose=True,
                 carousel_recommender_list = None):

        assert carousel_recommender_list is not None
        self.carousel_recommender_list = carousel_recommender_list


        super(EvaluatorMultipleCarousels, self).__init__(URM_test_list, cutoff_list,
                                                         diversity_object = diversity_object,
                                                         min_ratings_per_user =min_ratings_per_user, exclude_seen=exclude_seen,
                                                         ignore_items = ignore_items, ignore_users = ignore_users,
                                                         verbose = verbose)

        assert len(cutoff_list)==1
        self.cutoff_list = [cutoff_list[0]]
        self.cutoff_list_multiple = int(cutoff_list[0] * (len(self.carousel_recommender_list) + 1))





    def _run_evaluation_on_selected_users(self, recommender_object, users_to_evaluate, block_size = None):

        if block_size is None:
            block_size = min(1000, int(1e8/self.n_items))
            block_size = min(block_size, len(users_to_evaluate))


        results_dict = _create_empty_metrics_dict_carousel([self.cutoff_list_multiple],
                                                          self.n_items, self.n_users,
                                                          recommender_object.get_URM_train(),
                                                          self.URM_test,
                                                          self.ignore_items_ID,
                                                          self.ignore_users_ID,
                                                          self.diversity_object)


        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        # Start from -block_size to ensure it to be 0 at the first block
        user_batch_start = 0
        user_batch_end = 0

        while user_batch_start < len(users_to_evaluate):

            user_batch_end = user_batch_start + block_size
            user_batch_end = min(user_batch_end, len(users_to_evaluate))

            test_user_batch_array = np.array(users_to_evaluate[user_batch_start:user_batch_end])
            user_batch_start = user_batch_end

            recommended_items_batch_list_all_bands = None
            scores_batch_all_bands = None

            for recommender_band in [*self.carousel_recommender_list, recommender_object]:

                # Compute predictions for a batch of users using vectorization, much more efficient than computing it one at a time
                recommended_items_batch_list, scores_batch = recommender_band.recommend(test_user_batch_array,
                                                                          remove_seen_flag=self.exclude_seen,
                                                                          cutoff = self.max_cutoff,
                                                                          remove_top_pop_flag=False,
                                                                          remove_custom_items_flag=self.ignore_items_flag,
                                                                          return_scores = True
                                                                         )

                recommended_items_batch_list = np.array(recommended_items_batch_list)
                scores_batch = np.array(scores_batch)

                if recommended_items_batch_list_all_bands is None:
                    recommended_items_batch_list_all_bands = recommended_items_batch_list
                    scores_batch_all_bands = scores_batch
                else:

                    recommended_items_batch_list_all_bands = np.hstack((recommended_items_batch_list_all_bands, recommended_items_batch_list))
                    scores_batch_all_bands = np.hstack((scores_batch_all_bands, scores_batch))


            results_dict = self._compute_metrics_on_recommendation_list(test_user_batch_array = test_user_batch_array,
                                                         recommended_items_batch_list = recommended_items_batch_list_all_bands,
                                                         scores_batch = scores_batch,
                                                         results_dict = results_dict)


        # Restore correct cutoffs
        results_dict_correct_cutoff = {int(key / (len(self.carousel_recommender_list) + 1)):val for key, val in results_dict.items()}

        return results_dict_correct_cutoff




    def _compute_metrics_on_recommendation_list(self, test_user_batch_array, recommended_items_batch_list, scores_batch, results_dict):

        assert len(recommended_items_batch_list) == len(test_user_batch_array), "{}: recommended_items_batch_list contained recommendations for {} users, expected was {}".format(
            self.EVALUATOR_NAME, len(recommended_items_batch_list), len(test_user_batch_array))

        assert scores_batch.shape[0] == len(test_user_batch_array), "{}: scores_batch contained scores for {} users, expected was {}".format(
            self.EVALUATOR_NAME, scores_batch.shape[0], len(test_user_batch_array))

        assert scores_batch.shape[1] == self.n_items, "{}: scores_batch contained scores for {} items, expected was {}".format(
            self.EVALUATOR_NAME, scores_batch.shape[1], self.n_items)


        # Compute recommendation quality for each user in batch
        for batch_user_index in range(len(recommended_items_batch_list)):

            test_user = test_user_batch_array[batch_user_index]

            relevant_items = self.get_user_relevant_items(test_user)

            recommended_items = recommended_items_batch_list[batch_user_index]
            unique_recommended_items, unique_recommended_items_indices = np.unique(recommended_items, return_index=True)

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=False)
            is_relevant_and_unique = np.zeros_like(is_relevant)
            is_relevant_and_unique[unique_recommended_items_indices] = is_relevant[unique_recommended_items_indices]

            self._n_users_evaluated += 1

            results_current_cutoff = results_dict[self.cutoff_list_multiple]

            is_relevant_current_cutoff = is_relevant[0:self.cutoff_list_multiple]
            is_relevant_and_unique_current_cutoff = is_relevant_and_unique[0:self.cutoff_list_multiple]
            recommended_items_current_cutoff = recommended_items[0:self.cutoff_list_multiple]

            results_current_cutoff[EvaluatorMetrics.ROC_AUC.value]              += roc_auc(is_relevant_and_unique_current_cutoff)
            results_current_cutoff[EvaluatorMetrics.PRECISION.value]            += precision(is_relevant_and_unique_current_cutoff)
            results_current_cutoff[EvaluatorMetrics.PRECISION_RECALL_MIN_DEN.value]   += precision_recall_min_denominator(is_relevant_and_unique_current_cutoff, len(relevant_items))
            results_current_cutoff[EvaluatorMetrics.RECALL.value]               += recall(is_relevant_and_unique_current_cutoff, relevant_items)
            results_current_cutoff[EvaluatorMetrics.HIT_RATE.value]             += is_relevant_and_unique_current_cutoff.sum()
            results_current_cutoff[EvaluatorMetrics.ARHR.value]                 += arhr(is_relevant_and_unique_current_cutoff)
            results_current_cutoff[EvaluatorMetrics.MRR.value].add_recommendations(is_relevant_and_unique_current_cutoff)
            results_current_cutoff[EvaluatorMetrics.MAP.value].add_recommendations(is_relevant_and_unique_current_cutoff, relevant_items)

            # Set the duplicated occurrences of relevant items to None
            recommended_items_no_relevant_duplicates = list(recommended_items_current_cutoff)
            relevant_duplicates = np.logical_and(is_relevant_current_cutoff, np.logical_not(is_relevant_and_unique_current_cutoff))

            for relevant_duplicate_index in relevant_duplicates.nonzero()[0]:
                recommended_items_no_relevant_duplicates[relevant_duplicate_index] = None

            results_current_cutoff[EvaluatorMetrics.NDCG.value] += ndcg(recommended_items_no_relevant_duplicates, relevant_items,
                                                                        relevance=self.get_user_test_ratings(test_user))

            results_current_cutoff[CarouselEvaluatorMetrics.NDCG_2D.value] += ndcg_2d(recommended_items_current_cutoff, relevant_items,
                                                                                        relevance=self.get_user_test_ratings(test_user),
                                                                                        single_list_len = self.cutoff_list[0])

            results_current_cutoff[EvaluatorMetrics.NOVELTY.value].add_recommendations(recommended_items_current_cutoff)
            results_current_cutoff[EvaluatorMetrics.AVERAGE_POPULARITY.value].add_recommendations(recommended_items_current_cutoff)
            results_current_cutoff[EvaluatorMetrics.DIVERSITY_GINI.value].add_recommendations(recommended_items_current_cutoff)
            results_current_cutoff[EvaluatorMetrics.SHANNON_ENTROPY.value].add_recommendations(recommended_items_current_cutoff)
            results_current_cutoff[EvaluatorMetrics.COVERAGE_ITEM.value].add_recommendations(recommended_items_current_cutoff)
            results_current_cutoff[EvaluatorMetrics.COVERAGE_ITEM_CORRECT.value].add_recommendations(recommended_items_current_cutoff, is_relevant_current_cutoff)
            results_current_cutoff[EvaluatorMetrics.COVERAGE_USER.value].add_recommendations(recommended_items_current_cutoff, test_user)
            results_current_cutoff[EvaluatorMetrics.COVERAGE_USER_CORRECT.value].add_recommendations(is_relevant_current_cutoff, test_user)
            # results_current_cutoff[EvaluatorMetrics.DIVERSITY_MEAN_INTER_LIST.value].add_recommendations(recommended_items_current_cutoff)
            results_current_cutoff[EvaluatorMetrics.DIVERSITY_HERFINDAHL.value].add_recommendations(recommended_items_current_cutoff)

            results_current_cutoff[EvaluatorMetrics.RATIO_SHANNON_ENTROPY.value].add_recommendations(recommended_items_current_cutoff)
            results_current_cutoff[EvaluatorMetrics.RATIO_DIVERSITY_HERFINDAHL.value].add_recommendations(recommended_items_current_cutoff)
            results_current_cutoff[EvaluatorMetrics.RATIO_DIVERSITY_GINI.value].add_recommendations(recommended_items_current_cutoff)
            results_current_cutoff[EvaluatorMetrics.RATIO_NOVELTY.value].add_recommendations(recommended_items_current_cutoff)
            results_current_cutoff[EvaluatorMetrics.RATIO_AVERAGE_POPULARITY.value].add_recommendations(recommended_items_current_cutoff)

            if EvaluatorMetrics.DIVERSITY_SIMILARITY.value in results_current_cutoff:
                results_current_cutoff[EvaluatorMetrics.DIVERSITY_SIMILARITY.value].add_recommendations(recommended_items_current_cutoff)


        if time.time() - self._start_time_print > 30 or self._n_users_evaluated==len(self.users_to_evaluate):

            elapsed_time = time.time()-self._start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            self._print("Processed {} ({:4.1f}%) in {:.2f} {}. Users per second: {:.0f}".format(
                          self._n_users_evaluated,
                          100.0* float(self._n_users_evaluated)/len(self.users_to_evaluate),
                          new_time_value, new_time_unit,
                          float(self._n_users_evaluated)/elapsed_time))

            sys.stdout.flush()
            sys.stderr.flush()

            self._start_time_print = time.time()


        return results_dict


