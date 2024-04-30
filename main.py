import argparse
import openai
import pandas as pd
import random
import json
import os
import pickle    
import numpy as np  
def main():
    parser = argparse.ArgumentParser(description="A script for recommendation system")
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset to use')
    parser.add_argument('--recsys_model', type=str, required=True, help='Recommender model to use')
    #parser.add_argument('--language_model', type=str, required=True, help='Language model to use')

    args = parser.parse_args()

    dataset_name = args.dataset_name
    recsys_model = args.recsys_model
    #language_model = args.language_model

    from model_dump import model_dump
    model_dump(dataset_name, recsys_model)
      
    directory = f'results_{dataset_name}_{recsys_model}'
    directory_k = f'results_{dataset_name}_{recsys_model}_top_k'

if __name__ == "__main__":
    main()
