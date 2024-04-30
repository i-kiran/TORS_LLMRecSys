import argparse
import openai
import pandas as pd
import random
import json
import os
import pickle
import pandas as pd
import numpy as np
from recbole.config import Config
from recbole.model.general_recommender.bpr import BPR
from recbole.model.loss import BPRLoss
from recbole.model.general_recommender.neumf import NeuMF
from recbole.model.general_recommender.itemknn import ItemKNN
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger
from recbole.utils.case_study import full_sort_topk
import torch

import logging
from logging import getLogger
# For typing functions
from typing import List, Tuple
from collections import defaultdict
def model_dump(dataset_name, recsys_model):

    if recsys_model == 'BPR':
        from train_model import SVDModel
        if dataset_name == 'ML1M':
            config = Config(model='BPR', dataset='ml-1m', config_file_list=['your_config_file.yaml'])
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)
        elif dataset_name == 'ML100k':
             config = Config(model='BPR', dataset='ml-100k', config_file_list=['your_config_file.yaml'])
             dataset = create_dataset(config)
             train_data, valid_data, test_data = data_preparation(config, dataset)
        elif dataset_name == 'bookcrossing':
            config = Config(model='BPR', dataset='book-crossing', config_file_list=['your_config_file.yaml'])
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)
        model = BPR(config, train_data.dataset).to(config['device'])
        trainer = Trainer(config, model)
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data,show_progress=True)
        trainset = pd.DataFrame(train_data.dataset.inter_feat.numpy())
        testset = pd.DataFrame(test_data.dataset.inter_feat.numpy())
        ratings_df = pd.DataFrame(dataset.inter_feat.numpy())
        from recbole.data import Interaction
        test_item_ratings = testset.groupby('user_id').apply(lambda x: list(zip(x.item_id, x.rating))).to_dict()
        test_items = testset.groupby('user_id').apply(lambda x: list(x.item_id)).to_dict()
        pred_set = {}
        for key,value in test_items.items():
                tuple_list = []
                #index = key
                index = str(key)
                user_id = dataset.token2id(model.USER_ID, [index])[0]
                interaction = Interaction({model.USER_ID: [user_id]})
                predictions = model.full_sort_predict(interaction)
                predictions_list = predictions.tolist()
                for item, prediction in enumerate(predictions_list):
                        if item in value:
                            item_int = int(item)
                            for tuple in test_item_ratings[key]:
                                if tuple[0] == item:
                                    rating = tuple[1]
                                    break
                            tuple_list.append([item_int, rating, prediction])
                pred_set[key] = tuple_list
 
        trainset_k = dataset.inter_feat.numpy()

        all_movies = set(ratings_df['item_id'].unique())
        all_users = set(ratings_df['user_id'].unique())
        all_user_movie_pairs = pd.MultiIndex.from_product([all_users, all_movies], names=['user_id', 'item_id']).to_frame(index=False)
        testset_k = all_user_movie_pairs[~all_user_movie_pairs.isin(ratings_df)].dropna()
        testset_k['rating'] = 0

        test_item_ratings_k = testset_k.groupby('user_id').apply(lambda x: list(zip(x.item_id, x.rating))).to_dict()
        test_items_k = testset.groupby('user_id').apply(lambda x: list(x.item_id)).to_dict()
        pred_set_k = {}
        for key,value in test_items_k.items():
            tuple_list = []
            index = str(key)
            user_id = dataset.token2id(model.USER_ID, [index])[0]
            interaction = Interaction({model.USER_ID: [user_id]})
            predictions = model.full_sort_predict(interaction)
            predictions_list = predictions.tolist()
            for item, prediction in enumerate(predictions_list):
                if item in value:
                    item_int = int(item)
                    for tuple in test_item_ratings[key]:
                            if tuple[0] == item:
                                    rating = tuple[1]
                                    break
                    tuple_list.append([item_int, rating, prediction])
            pred_set_k[key] = tuple_list
        from save_results import save_results
        save_results(ratings_df, trainset, testset, pred_set ,dataset_name, recsys_model, trainset_k, testset_k, pred_set_k)

    elif recsys_model == 'NeuMF':
        from train_model import SVDModel
        if dataset_name == 'ML1M':
            config = Config(model='NeuMF', dataset='ml-1m', config_file_list=['your_config_file.yaml'])
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)       
        elif dataset_name == 'ML100k':
            config = Config(model='NeuMF', dataset='ml-100k', config_file_list=['your_config_file.yaml'])
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)
        elif dataset_name == 'bookcrossing':
            config = Config(model='NeuMF', dataset='book-crossing', config_file_list=['your_config_file.yaml'])
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)
        model = NeuMF(config, train_data.dataset).to(config['device'])
        trainer = Trainer(config, model)
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=True)
        trainset = pd.DataFrame(train_data.dataset.inter_feat.numpy())
        testset = pd.DataFrame(test_data.dataset.inter_feat.numpy())
        ratings_df = pd.DataFrame(dataset.inter_feat.numpy())
        from recbole.data import Interaction
        test_item_ratings = testset.groupby('user_id').apply(lambda x: list(zip(x.item_id, x.rating))).to_dict()
        test_items = testset.groupby('user_id').apply(lambda x: list(x.item_id)).to_dict()
        pred_set = {}
        i = 0
        pred_set = {}
        for key,value in test_item_ratings.items():
            i += 1
            index = str(key)
            user_id = dataset.token2id(model.USER_ID, [index])[0]
            item_ids = []
            true_ratings = []
            for tuple in value:
                item_index = str(tuple[0])
                true_rating = tuple[1]
                try:
                    item_id = dataset.token2id(model.ITEM_ID, [item_index])[0]
                    item_ids.append(item_id)
                    true_ratings.append(true_rating)
                except ValueError:
                    continue

            if item_ids:
                    interactions = Interaction({model.USER_ID: [user_id]*len(item_ids), model.ITEM_ID: item_ids})
                    predictions = model.predict(interactions)
                    list_pred = []
                    for item_id, prediction, true_rating in zip(item_ids, predictions, true_ratings):
                        list_pred.append([item_id, true_rating, prediction.item()])
            pred_set[user_id] = list_pred 
        trainset_k = dataset.inter_feat.numpy()

        all_movies = set(ratings_df['item_id'].unique())
        all_users = set(ratings_df['user_id'].unique())
        all_user_movie_pairs = pd.MultiIndex.from_product([all_users, all_movies], names=['user_id', 'item_id']).to_frame(index=False)
        testset_k = all_user_movie_pairs[~all_user_movie_pairs.isin(ratings_df)].dropna()
        testset_k['rating'] = 0

        test_item_ratings_k = testset_k.groupby('user_id').apply(lambda x: list(zip(x.item_id, x.rating))).to_dict()
        test_items_k = testset.groupby('user_id').apply(lambda x: list(x.item_id)).to_dict()
        pred_set_k = {}
        for key,value in test_items_k.items():
                    index = str(key)
                    user_id = dataset.token2id(model.USER_ID, [index])[0]
                    item_ids = []
                    for item in value:
                        item_str = str(item)
                        try:
                            item_id = dataset.token2id(model.ITEM_ID, [item_str])[0]
                            item_ids.append(item_id)
                        except ValueError:
                            continue
                    if item_ids:  
                        interactions = Interaction({model.USER_ID: [user_id]*len(item_ids), model.ITEM_ID: item_ids})
                        predictions = model.predict(interactions)
                        for item_id, prediction in zip(item_ids, predictions):
                            pred_set_k[(user_id, item_id)] = prediction

        pred_set_new_k = {}
        for key in pred_set_k:
            prediction = pred_set_k[key]
            if prediction.numel() == 1: 
                pred_set_new_k[key] = prediction.item()

        data_new = [(user_id, item_id, rating) for (user_id, item_id), rating in pred_set_new_k.items()]
        pred_set_2 = pd.DataFrame(data_new, columns=['user_id', 'item_id', 'rating'])
        from save_results import save_results
        save_results(ratings_df, trainset, testset, pred_set ,dataset_name, recsys_model, trainset_k, testset_k, pred_set_2)

    elif recsys_model == 'ItemKNN':
        if dataset_name == 'ML1M':
            config = Config(model='ItemKNN', dataset='ml-1m', config_file_list=['your_config_file.yaml'])
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)       
        elif dataset_name == 'ML100k':
            config = Config(model='ItemKNN', dataset='ml-100k', config_file_list=['your_config_file.yaml'])
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)
        elif dataset_name == 'bookcrossing':
            config = Config(model='ItemKNN', dataset='book-crossing', config_file_list=['your_config_file.yaml'])
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)
        model = ItemKNN(config, train_data.dataset).to(config['device'])
        trainer = Trainer(config, model)
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=True)
        trainset = pd.DataFrame(train_data.dataset.inter_feat.numpy())
        testset = pd.DataFrame(test_data.dataset.inter_feat.numpy())
        ratings_df = pd.DataFrame(dataset.inter_feat.numpy())
        from recbole.data import Interaction
        test_item_ratings = testset.groupby('user_id').apply(lambda x: list(zip(x.item_id, x.rating))).to_dict()
        test_items = testset.groupby('user_id').apply(lambda x: list(x.item_id)).to_dict()
        pred_set = {}
        i = 0
        pred_set = {}
        for key,value in test_item_ratings.items():
            i += 1
            index = str(key)
            user_id = dataset.token2id(model.USER_ID, [index])[0]
            item_ids = []
            true_ratings = []
            for tuple in value:
                item_index = str(tuple[0])
                true_rating = tuple[1]
                try:
                    item_id = dataset.token2id(model.ITEM_ID, [item_index])[0]
                    item_ids.append(item_id)
                    true_ratings.append(true_rating)
                except ValueError:
                    continue

            if item_ids:
                    interactions = Interaction({model.USER_ID: [user_id]*len(item_ids), model.ITEM_ID: item_ids})
                    predictions = model.predict(interactions)
                    list_pred = []
                    for item_id, prediction, true_rating in zip(item_ids, predictions, true_ratings):
                        list_pred.append([item_id, true_rating, prediction.item()])
            pred_set[user_id] = list_pred
        trainset_k = dataset.inter_feat.numpy()

        all_movies = set(ratings_df['item_id'].unique())
        all_users = set(ratings_df['user_id'].unique())
        all_user_movie_pairs = pd.MultiIndex.from_product([all_users, all_movies], names=['user_id', 'item_id']).to_frame(index=False)
        testset_k = all_user_movie_pairs[~all_user_movie_pairs.isin(ratings_df)].dropna()
        testset_k['rating'] = 0

        test_item_ratings_k = testset_k.groupby('user_id').apply(lambda x: list(zip(x.item_id, x.rating))).to_dict()
        test_items_k = testset.groupby('user_id').apply(lambda x: list(x.item_id)).to_dict()
        pred_set_k = {}
        for key,value in test_items_k.items():
                    index = str(key)
                    user_id = dataset.token2id(model.USER_ID, [index])[0]
                    item_ids = []
                    for item in value:
                        item_str = str(item)
                        try:
                            item_id = dataset.token2id(model.ITEM_ID, [item_str])[0]
                            item_ids.append(item_id)
                        except ValueError:
                            continue
                    if item_ids: 
                        interactions = Interaction({model.USER_ID: [user_id]*len(item_ids), model.ITEM_ID: item_ids})
                        predictions = model.predict(interactions)
                        for item_id, prediction in zip(item_ids, predictions):
                            pred_set_k[(user_id, item_id)] = prediction

        pred_set_new_k = {}
        for key in pred_set_k:
            prediction = pred_set_k[key]
            if prediction.numel() == 1: 
                pred_set_new_k[key] = prediction.item()

        data_new = [(user_id, item_id, rating) for (user_id, item_id), rating in pred_set_new_k.items()]
        pred_set_2 = pd.DataFrame(data_new, columns=['user_id', 'item_id', 'rating'])
        from save_results import save_results
        save_results(ratings_df, trainset, testset, pred_set ,dataset_name, recsys_model, trainset_k, testset_k, pred_set_2)

if __name__ == "__main__":
    model_dump()
