
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.metrics import roc_auc_score
from catboost import datasets, Pool
from catboost.utils import get_roc_curve, eval_metric
import numpy as np
from sklearn.metrics import ndcg_score
def get_user_ratings(predictions, test_df):
    user_ratings = {}
    users = predictions.keys()
    for key in users:
        true_r = []
        est = []
        for value in predictions[key]:
            uid = key
            rating_list = value
            print(rating_list)
            true_r_val = rating_list[1]
            est_val = rating_list[2]
            true_r.append(true_r_val)
            est.append(est_val)
        user_ratings[uid] = [true_r,est]
    return user_ratings

def calculate_auc_scores(user_ratings):
    auc_scores = {}
    user_ids = list(user_ratings.keys())
    for user_id in user_ids:
        auc_score = eval_metric(user_ratings[user_id][0],user_ratings[user_id][1], 'AUC:type=Ranking')[0]
        auc_scores[user_id] = auc_score

    sorted_auc_scores = dict(sorted(auc_scores.items(), key=lambda item: item[1], reverse=True))
    return sorted_auc_scores


def calculate_ndcg_scores(user_ratings):
    ndcg_scores = {}
    user_ids = list(user_ratings.keys())
    for user_id in user_ids:
        if len(user_ratings[user_id][1]) <= 1 or len(user_ratings[user_id][0]) <= 1:
            ndcg_scores[user_id] = 0
            continue
        true_relevance = np.asarray([np.array(user_ratings[user_id][1]) + abs(min(user_ratings[user_id][1]))])
        scores = np.asarray([user_ratings[user_id][0]])
        ndcg_scores[user_id] = ndcg_score(true_relevance, scores, k=20)

    sorted_ndcg_scores = dict(sorted(ndcg_scores.items(), key=lambda item: item[1], reverse=True))
    return sorted_ndcg_scores

def calculate_sparsity_scores(ratings_df, test_df):
    pivot_table = ratings_df.pivot(index='user_id', columns='item_id', values='rating')
    sparsity_values = {}
    # Iterate over the users in the new test set
    for user_id in test_df['user_id'].unique():
        user_ratings = pivot_table.loc[user_id]
        sparsity = user_ratings.isnull().sum() / len(user_ratings)
        sparsity_values[user_id] = sparsity

    sorted_sparsity_scores = dict(sorted(sparsity_values.items(), key=lambda item: item[1], reverse=True))
    return sorted_sparsity_scores

def calculate_metrics(testset, predictions, ratings_df):
    test_df = testset
    print("finished converting testset to dataframe")
    #count the number of ratings each user has made
    user_rating_counts = test_df['user_id'].value_counts()

    min_ratings = user_rating_counts.min()
    max_ratings = user_rating_counts.max()
    user_dict = get_user_ratings(predictions, testset)

    sorted_auc_scores = calculate_auc_scores(user_dict)
    sorted_ndcg_scores = calculate_ndcg_scores(user_dict)
    return  user_dict,sorted_auc_scores, sorted_sparsity_scores, sorted_ndcg_scores
