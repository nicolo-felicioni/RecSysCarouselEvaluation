# Carousel Evaluation for Recommender Systems

This is the repository with the code for the paper "A Methodology for the Offline Evaluation of RecommenderSystems in a User Interface with Multiple Carousels", published at UMAP Late-Breaking Results 2021.

For information on the requirements and how to install this repository, see the following [Installation](#Installation) section, for information on the structure of the repo and the recommender models see the [Project structure](#Project-structure) section.
Istructions on how to run the experiments are in section [Run the experiments](#Run-the-experiments).
The additional material, with additional experiments, is in the file [additional_material.pdf](additional_material.pdf).

## Installation

Note that this repository requires Python 3.6

First we suggest you create an environment for this project using Anaconda.

First checkout this repository, then enter in the repository folder and run this commands to create and activate a new environment:

If you are using conda:
```Python
conda create -n CarouselEvaluation python=3.6 anaconda
conda activate CarouselEvaluation
```

Then install all the requirements and dependencies
```Python
pip install -r requirements.txt
```

In order to compile you must have installed: _gcc_ and _python3 dev_, which can be installed with the following commands:
```Python
sudo apt install gcc 
sudo apt-get install python3-dev
```

At this point you can compile all Cython algorithms by running the following command. The script will compile within the current active environment. The code has been developed for Linux and Windows platforms. During the compilation you may see some warnings. 
 
```Python
python run_compile_all_cython.py
```



## Project structure

#### Evaluation
The Evaluator class is used to evaluate a recommender object. It computes various metrics:
* Accuracy metrics: ROC_AUC, PRECISION, RECALL, MAP, MRR, NDCG, F1, HIT_RATE, ARHR
* Beyond-accuracy metrics: NOVELTY, DIVERSITY, COVERAGE

The evaluator takes as input the URM against which you want to test the recommender, then a list of cutoff values (e.g., 5, 20) and, if necessary, an object to compute diversity.
The function evaluateRecommender will take as input only the recommender object you want to evaluate and return both a dictionary in the form {cutoff: results}, where results is {metric: value} and a well-formatted printable string.

```python

    from Evaluation.Evaluator import EvaluatorHoldout

    evaluator_test = EvaluatorHoldout(URM_test, [5, 20])

    results_run_dict, results_run_string = evaluator_test.evaluateRecommender(recommender_instance)

    print(results_run_string)

```


### Recommenders
Contains some basic modules and the base classes for different Recommender types.
All recommenders inherit from BaseRecommender, therefore have the same interface.
You must provide the data when instantiating the recommender and then call the _fit_ function to build the corresponding model.

Each recommender has a _compute_item_score function which, given an array of user_id, computes the prediction or _score_ for all items.
Further operations like removing seen items and computing the recommendation list of the desired length are done by the _recommend_ function of BaseRecommender

As an example:

```python
    user_id = 158
    
    recommender_instance = ItemKNNCFRecommender(URM_train)
    recommender_instance.fit(topK=150)
    recommended_items = recommender_instance.recommend(user_id, cutoff = 20, remove_seen_flag=True)
    
    recommender_instance = SLIM_ElasticNet(URM_train)
    recommender_instance.fit(topK=150, l1_ratio=0.1, alpha = 1.0)
    recommended_items = recommender_instance.recommend(user_id, cutoff = 20, remove_seen_flag=True)
```

The similarity module allows to compute the item-item or user-user similarity.
It is used by calling the Compute_Similarity class and passing which is the desired similarity and the sparse matrix you wish to use.

It is able to compute the following similarities: Cosine, Adjusted Cosine, Jaccard, Tanimoto, Pearson and Euclidean (linear and exponential)

```python

    similarity = Compute_Similarity(URM_train, shrink=shrink, topK=topK, normalize=normalize, similarity = "cosine")

    W_sparse = similarity.compute_similarity()

```




## Run the experiments

See see the following [Installation](#Installation) section for information on how to install this repository.
After the installation is complete you can run the experiments.

* Run the 'run_hyperparameter_optimization.py' script to perform the hyperparameter optimization of the various algorithms independently. In the script you may select which of the datasets to use. The script will automatically download the data (only possible for MovieLens10M. For the Netflix and Spotify2018 dataset you will have to download it from the link that will be prompted) and save the optimized models.
* Run the 'run_evaluation_carousel.py' to evaluate the previously optimized models under a carousel setting where the first carousel is a TopPop model.



## Acknowledgement
This repository is baded upon this project [this project](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation), released under a AGPL-3.0 License.
