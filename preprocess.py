#imports 
import pandas as pd
import numpy as np

#load data
def load_data():
    ratings_cols = ['user_id', 'item_id', 'rating', 'unix_timestamp']
    users_cols = ['user_id', 'sex', 'age', 'occupation', 'zip_code']
    
    items_cols = ['item_id', 'title' ,'generes']
    ratings_df = pd.read_csv('/Users/kirandeepkaur/Downloads/RECOMMENDERS/Final_Exp/Recbole/Datasets/ml-1m/ml-1m/ratings.dat', sep='::', names=ratings_cols, encoding='latin-1', engine='python')
    users_df = pd.read_csv('/Users/kirandeepkaur/Downloads/RECOMMENDERS/Final_Exp/Recbole/Datasets/ml-1m/ml-1m/users.dat', sep='::', names=users_cols, encoding='latin-1', engine='python')
    items_df = pd.read_csv('/Users/kirandeepkaur/Downloads/RECOMMENDERS/Final_Exp/Recbole/Datasets/ml-1m/ml-1m/movies.dat', sep='::', names=items_cols, encoding='latin-1', engine='python')
    return ratings_df, users_df, items_df

