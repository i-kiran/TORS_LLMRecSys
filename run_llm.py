import argparse
import pandas as pd
import random
import json
import os
import openai
import pickle    
import numpy as np  
from collections import defaultdict
from llamaapi import LlamaAPI
from sklearn.metrics import ndcg_score
import numpy as np

def generate_ranklists(predictions):
            # First map the predictions to each user.
            top_n = defaultdict(list)
            for uid, iid, true_r, est in predictions.itertuples(index=False):
                top_n[uid].append((iid, est))
    
            # Then sort the predictions for each user and retrieve the k highest ones.
            for uid, user_ratings in top_n.items():
                #print(uid , user_ratings)
                user_ratings.sort(key=lambda x: x[1], reverse=True)
                top_n[uid] = user_ratings
            top_n
            return top_n
def generate_responses_GPT(prompts):
        # Initialize a list to store the responses
        responses = []
        openai.api_key = 'your_api'
        # Iterate over the prompts
        for prompt in prompts:
            # Generate a response to the prompt
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                temperature = 0,
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": "You are a very smart user preference based recommendation system designed to output JSON."},
                    {"role": "user", "content": prompt}
                ])
            # Append the response to the list
            print(response.choices[0].message.content)
            responses.append(response.choices[0].message.content)
        print("finished")
        return responses
def generate_responses_Mixtral(prompts):
      responses = []
      llama = LlamaAPI('LL-dTNEB2i1WDhLuc2KNGkLtDwsmKgo3rgSIxKJSHKio7BazzGUGLh98JFltm06LKTO')
      for prompt in prompts:
          api_request_json = {
                              "model": "mixtral-8x7b-instruct",
                              "messages": [
                                  {"role": "system", "content": "You are a movie recommendation agent that ranks the randomly given movies in correct order. Based on the given preferences of the user, your job is to rank the candidate movies according to the preferences demonstrated by user. In the output dictionary only add user_id and ranked list of items in the form of python dictionary."},
                                  {"role": "user", "content": prompt},
                              ],
                              "temperature":1,
                              }
          response = llama.run(api_request_json)
          print(json.dumps(response.json(), indent=2))
          responses.append(json.dumps(response.json()))
def parse_responses(loaded_responses, items_dict):
        '''
        This method parses the responses, which are assumed to be JSON strings. 
        Each response is expected to contain a 'user_id' and 'recommended_items'. 
        The method returns two dictionaries: user_recommendations (mapping user IDs
          to lists of recommended items) and user_recommendations_new (mapping user 
          IDs to lists of tuples, where each tuple contains an item and its rank).
        '''
        user_recommendations = {}
        user_recommendations_new = {}
        items_dict_reversed = {v: k for k, v in items_dict.items()}
        for i in range(len(loaded_responses)):
            #for response in loaded_responses:
            print(i)
            # Parse the JSON response
            data = json.loads(loaded_responses[i])
            print(data)
            # Check if the necessary keys exist in the response
            if 'user_id' in data and 'ranked_list_of_items'in data:
                # Extract the user_id and recommended_movies
                user_id = data['user_id']
                recommended_movies = data['ranked_list_of_items']
                recommended_movies = [items_dict_reversed[item] for item in recommended_movies if item in items_dict_reversed]
                # Add to the dictionary
                user_recommendations[user_id] = recommended_movies
                # Assign ranks to the recommended movies
                ranked_movies_responses = [(item_id, len(recommended_movies) - rank + 1) for rank, item_id in enumerate(recommended_movies, start=1)]
                # Add to the dictionary
                user_recommendations_new[user_id] = ranked_movies_responses
        print("Done parsing responses and assigning ranks")
        return user_recommendations, user_recommendations_new
        
def main():
    parser = argparse.ArgumentParser(description="A script for recommendation system")
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset to use')
    parser.add_argument('--recsys_model', type=str, required=True, help='Recommender model to use')
    parser.add_argument('--language_model', type=str, required=True, help='Language model to use')

    args = parser.parse_args()

    dataset_name = args.dataset_name
    recsys_model = args.recsys_model
    language_model = args.language_model
    directory = f'results_{dataset}_{recsys_model}'
    directory_k = f'results_{dataset}_{recsys_model}_top_k'
    testset_file = os.path.join(directory, 'testset.pkl')
    trainset_file = os.path.join(directory, 'trainset.pkl')
    predictions_file = os.path.join(directory, 'predictions.pkl')
    ratings_df_file = os.path.join(directory, 'ratings_df.pkl')
    testset_file_k = os.path.join(directory_k, 'testset_k.pkl') #all the unrated items for each user
    trainset_file_k = os.path.join(directory_k, 'trainset_k.pkl') #all the rated items for each user#
    predictions_file_k = os.path.join(directory_k, 'predictions_k.pkl') # ranked items for each user
    with open(trainset_file, 'rb') as f:
        trainset = pickle.load(f)
    with open(testset_file, 'rb') as f:
            testset = pickle.load(f)
    with open(predictions_file, 'rb') as f:
            predictions = pickle.load(f)
    with open(ratings_df_file, 'rb') as f:
            ratings_df = pickle.load(f)

    from eval_RS import calculate_metrics
    u_d,sorted_auc_scores, sorted_sparsity_scores, ndcg_scores =   calculate_metrics(testset, predictions, ratings_df)

    from user_decision import plot_sparsity_vs_auc
    weak_users = plot_sparsity_vs_auc(sorted_sparsity_scores, sorted_auc_scores)
    auc_scores = list(sorted_auc_scores.values())
    sparsity_scores = list(sorted_sparsity_scores.values())
    average_auc_all_users = np.mean(auc_scores)
    average_sparsity = np.mean(sparsity_scores)
    auc_weak_users = [auc for user_id, auc in sorted_auc_scores.items() if auc <= 0.5 and sorted_sparsity_scores[user_id] > average_sparsity]
    ndcg_weak_users = [score for user_id, score in ndcg_scores.items() if score < 0.8]
    average_ndcg_weak_population = np.mean(ndcg_weak_users)
    average_auc_weak_population = np.mean(auc_weak_users)
    total_ndcg = sum(ndcg_scores.values())
    average_ndcg = sum(ndcg_scores.values()) / len(ndcg_scores)
    with open(trainset_file_k, 'rb') as f:
         trainset_k = pickle.load(f)
    file_path = os.path.join(directory_k, 'predictions_k.csv')
    df_predictions_k = pd.read_csv(file_path)

    df_predictions_k = df_predictions_k.rename(columns={'rating': 'prediction'})
    df_predictions_k['rating'] = 0
    df_predictions_k = df_predictions_k[['user_id', 'item_id', 'rating', 'prediction']]

    rank = generate_ranklists(df_predictions_k)

    weak_users_list = weak_users[['user_id']].values.tolist()
    weak_users_list = [item for sublist in weak_users_list  for item in sublist]
    from preprocess import load_data 
    ratings_df, users_df, items_df =load_data()

    items_dict = items_df.set_index('item_id')['title'].to_dict()
    for key in items_dict:
        items_dict[key] = ' '.join(items_dict[key].split()[:-1])
    trainset_df = pd.DataFrame(trainset, columns=['user_id', 'item_id', 'rating'])


    from construct_prompt import PromptGen
    prompt_generator = PromptGen(trainset_df, rank, weak_users_list, items_dict)
    prompts,ranked_test_set = prompt_generator.run()

    if language_model == 'GPT-3.5':
      responses = generate_responses_GPT
      filename = f'responses_{dataset}_{recsys_model}_GPT3.5.pkl'
      if os.path.exists(filename) and os.path.getsize(filename) > 0:
          # Open the file and load the pickled data
          with open(filename, 'rb') as f:
              loaded_responses = pickle.load(f)
      else:
          print("File is empty or does not exist.")
      user_recommendations, user_recommendations_new = parse_responses(loaded_responses, items_dict)
      # Initialize a dictionary to store the results
      user_items = {}
      
      # Create a reverse dictionary where keys are item names and values are item ids
      items_dict_reverse = {v: k for k, v in items_dict.items()}
      
      # Iterate over each user in ranked_test_set
      for user_id, item_names in ranked_test_set.items():
          # Initialize a list to store the item ids for this user
          item_ids = []
          item_names = item_names[:50]
          # Iterate over each item name
          for item_name in item_names:
              # If the item name is in items_dict_reverse, get the item id
              if item_name in items_dict_reverse:
                  item_id = items_dict_reverse[item_name]
                  # Add the item id to the list
                  item_ids.append(item_id)
          # Add the list of item ids to the user_items dictionary with the user_id as the key
          user_items[user_id] = item_ids

          # Create a reverse dictionary where keys are item names and values are item ids
          items_dict_reverse = {v: k for k, v in items_dict.items()}
          
          # Initialize a dictionary to store the results
          user_items_with_ranks = {}
          
          # Iterate over each user in ranked_test_set
          for user_id, item_names in ranked_test_set.items():
              # Initialize a list to store the item ids for this user
              item_ids = []
              item_names = item_names[:50]
              # Iterate over each item name
              for rank, item_name in enumerate(item_names, start=1):
                  # If the item name is in items_dict_reverse, get the item id
                  if item_name in items_dict_reverse:
                      item_id = items_dict_reverse[item_name]
                      # Add the item id and its rank to the list
                      item_ids.append((item_id, len(item_names) - rank + 1))
              # Add the list of item ids to the user_items_with_ranks dictionary with the user_id as the key
              user_items_with_ranks[user_id] = item_ids
              Original_Ranks = user_items_with_ranks
              New_ranks = user_recommendations_new
              
              new_dict = {}
              
              for user_id in New_ranks:
                  print(user_id)
                  if user_id in Original_Ranks.keys():
                      print(user_id)
                      new_dict[user_id] = []
                      for tuple_new in New_ranks[user_id]:
                          #print(tuple_new)
                          item_id_new, rank_new = tuple_new
                          for tuple_orig in Original_Ranks[user_id]:
                              item_id_orig, rank_orig = tuple_orig
                              if item_id_new == item_id_orig:
                                  #print( item_id_new, rank_orig, rank_new)
                                  new_dict[user_id].append([item_id_new, rank_orig, rank_new])
                  else:
                      print("No")


                  auc_list = []
                  for user_id in new_dict:
                      original_ranks = []
                      LLM_ranks = []
                      for item in new_dict[user_id]:
                          #print(item)
                          item_id, original_rank, reranked_rank = item
                          LLM_ranks.append(reranked_rank)
                          original_ranks.append(original_rank)
                      print(original_ranks)
                      print(LLM_ranks)
                      eval_metric_result = eval_metric(original_ranks,LLM_ranks, 'AUC:type=Ranking')
                      print(eval_metric_result)
                      auc_list.append(eval_metric_result

                      print(f"New_AUC_LLM:{np.mean(auc_list)}")

          new_dict_str_keys = {str(key): value for key, value in new_dict.items()}
          user_ids_weak = new_dict_str_keys.keys()
          user_auc_dict_weak_LLM = {user_id: auc for user_id, auc in zip(user_ids_weak, auc_list)}
          for key, value in sorted_auc_scores.items():
              s_k = str(key)
              if s_k in user_ids_weak and s_k in user_auc_dict_weak_LLM.keys():
                  print('yay')
                  sorted_auc_scores[key] = user_auc_dict_weak_LLM[s_k][0]
          
          list_x = []
          for key,value in sorted_auc_scores.items():
              list_x.append(value)
          
          np.mean(list_x)
          from sklearn.metrics import ndcg_score
          import numpy as np
          
          ndcg_list = []
          for user_id in new_dict:
              original_ranks = []
              LLM_ranks = []
              for item in new_dict[user_id]:
                  item_id, original_rank, reranked_rank = item
                  LLM_ranks.append(reranked_rank)
                  original_ranks.append(original_rank)
              
              # Skip if there are no items for this user
              if not original_ranks or not LLM_ranks:
                  continue
              
              # Convert lists to 2D arrays
              true_relevance = np.asarray([original_ranks])
              scores = np.asarray([LLM_ranks])
              
              # Calculate NDCG
              ndcg = ndcg_score(true_relevance, scores, k=20)
              ndcg_list.append(ndcg)
            print(f"Improved_NDCG:{np.mean(ndcg_list)}")

if __name__ == "__main__":
    main()
