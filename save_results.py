
import os
import pickle
import pandas as pd
def save_results(ratings_df, trainset, testset, predictions,dataset, recsys_model,trainset_k, testset_k, predictions_k):
    directory = f'results_{dataset}_{recsys_model}'
    directory_k = f'results_{dataset}_{recsys_model}_top_k'
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory_k):
        os.makedirs(directory_k)

    # Saving results
    with open(os.path.join(directory, 'trainset.pkl'), 'wb') as f:
        pickle.dump(trainset, f)
    with open(os.path.join(directory, 'testset.pkl'), 'wb') as f:
        pickle.dump(testset, f)
    with open(os.path.join(directory, 'predictions.pkl'), 'wb') as f:
        pickle.dump(predictions, f)
    with open(os.path.join(directory, 'ratings_df.pkl'), 'wb') as f:
        pickle.dump(ratings_df, f)
    with open(os.path.join(directory_k, 'trainset_k.pkl'), 'wb') as f:
        pickle.dump(trainset_k, f)
    with open(os.path.join(directory_k, 'testset_k.pkl'), 'wb') as f:
        pickle.dump(testset_k, f)
    # Initialize an empty list for the data
    data = []
    if recsys_model == 'NeuMF' or recsys_model == 'ItemKNN':
        predictions_k.to_csv(os.path.join(directory_k, 'predictions_k.csv'), index=False)
    else:
        for user_id, items in predictions_k.items():
            for item_id, true_rating, estimated_rating in items:
                data.append({'user_id': user_id, 'item_id': item_id, 'true_rating': true_rating, 'estimated_rating': estimated_rating})

            # Convert the list to a DataFrame
            df_predictions_k = pd.DataFrame(data)
            df_predictions_k.to_csv(os.path.join(directory_k, 'predictions_k.csv'), index=False)
    print(f"Results saved in {directory}")
