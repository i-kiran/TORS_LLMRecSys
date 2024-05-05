#from train_model import trainset, testset
import random
import pandas as pd


class PromptGen:
    def __init__(self, trainset, rank, weak_users_list,items_dict):
        self.trainset = trainset
        self.retreived_items = rank
        self.weak_users = weak_users_list
        self.items_dict = items_dict

    def convert_to_dataframe(self):
        '''
        This function is converting a pandas DataFrame trainset into another DataFrame df_train
        '''
        ratings = []
        for index, row in self.trainset.iterrows():
            raw_user_id = row['user_id']
            raw_item_id = row['item_id']
            rating = row['rating']
            ratings.append([raw_user_id, raw_item_id, rating])
        df_train = pd.DataFrame(ratings, columns=['user_id', 'item_id', 'rating'])
        print(df_train)
        return df_train
    
    def prepare_dataset(self, df_train):
        #print("preparing dataset")
        '''
        this function is preparing a dataset where each row corresponds to a user and
        contains the items and their ratings from the train set and the items from the 
        test set for that user.
        '''
        user_ids = []
        train_items = []
        test_items = []

        # Convert df_train to a dictionary for faster access
        df_train_dict = df_train.groupby('user_id')[['item_id', 'rating']].apply(lambda g: g.values.tolist()).to_dict()

        # Convert retreived_items to a dictionary for faster access
        retreived_items_dict = {k: [sublist[:2] for sublist in v] for k, v in self.retreived_items.items()}
        #print(self.weak_users)
        for user_id in self.weak_users:
            #print("user_id",user_id)
            # Find the items and their ratings that the user rated in the train set
            train_items_user = df_train_dict.get(user_id, [])
            
            # Find the items and their ratings that the user rated in the test set
            #test_items_user = retreived_items_dict.get(user_id, [])
            all_item_ids = set(df_train['item_id'].unique())
            rated_item_ids = set(df_train[df_train['user_id'] == user_id]['item_id'].unique())
            test_items_user = list(all_item_ids - rated_item_ids)
            #print("test_items_user",test_items_user)
            # Append the data to the lists
            user_ids.append(user_id)
            train_items.append(train_items_user)
            test_items.append(test_items_user)

        # Create a new DataFrame with the data
        df_items = pd.DataFrame({
            'user_id': user_ids,
            'train_items': train_items,
            'test_items': test_items
        })
        return df_items

    def sort_items_by_rating(self, df_items):
        # Define a function to sort a list of tuples by the second element in descending order
        def sort_by_rating(items):
            for i, item in enumerate(items):
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    print(f"Invalid item at index {i}: {item}")
                    continue
                return sorted(items, key=lambda x: x[1], reverse=True)

        # Sort the tuples in the train_items and test_items columns by rating
        df_items['train_items'] = df_items['train_items'].apply(sort_by_rating)
        #df_items['test_items'] = df_items['test_items'].apply(sort_by_rating)

        return df_items

    def extract_item_ids(self, df_items):
        # Define a function to extract item ids from a list of tuples
        def extract_ids(items):
            return [item_id for item_id, rating in items]

        # Extract the item ids from the tuples in the train_items and test_items columns
        df_items['train_items'] = df_items['train_items'].apply(extract_ids)
        #df_items['test_items'] = df_items['test_items'].apply(extract_ids)

        return df_items
    
    def replace_ids_with_titles(self, df_items, items_dict):
        # Define a function to replace item ids with titles
        def replace_ids(items):
            return [items_dict[item_id] for item_id in items if item_id in items_dict]

        # Replace the item ids with titles in the train_items and test_items columns
        df_items['train_items'] = df_items['train_items'].apply(replace_ids)
        df_items['test_items'] = df_items['test_items'].apply(replace_ids)

        return df_items

    def generate_prompts(self,df_items):
        # Initialize a list to store the prompts
        prompts = []
        sorted_testing_list = {}
        # Iterate over the rows in the DataFrame
        for _, row in df_items.iterrows():
            # Randomly select the top 20 items atleast from the train_items and test_items columns
            train_items = random.sample(row['train_items'][:20], min(20, len(row['train_items'])))
            test_items = random.sample(row['test_items'][:20], min(20, len(row['test_items'])))
            sorted_testing_list[row['user_id']] = test_items
            random.shuffle(test_items)
            # Format the prompt
            prompt = f"User {row['user_id']} rated following items in decreasing order of preference, where the topmost movie name is the most preferred one{train_items}. Based on these preferences rank following  candidate items in the decreasing order of preference such that item on top should be the most liked one {test_items}.Do not generate any movie name outside this candidate list."
            # Append the prompt to the list
            prompts.append(prompt)
        return prompts,sorted_testing_list

    def run(self):
        df_train = self.convert_to_dataframe()
        df_items = self.prepare_dataset(df_train)
        df_items = self.sort_items_by_rating(df_items)
        df_items = self.extract_item_ids(df_items)
        df_items = self.replace_ids_with_titles(df_items, self.items_dict)
        prompts,sorted_testing_list = self.generate_prompts(df_items)
        return prompts,sorted_testing_list
