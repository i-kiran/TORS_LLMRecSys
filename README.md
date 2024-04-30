# Efficient-and-Responsible-Adaptation-of-Large-Language-Models-for-Robust-Top-k-Recommendations
1. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```
2. Download and convert datasets to atomic files.
   ```bash
           git clone https://github.com/RUCAIBox/RecDatasets
            cd RecDatasets/conversion_tools
            pip install -r requirements.txt
            wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
            unzip ml-100k -d ml-100k
            wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
            unzip ml-1m.zip -d ml-1m
            wget http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip
            unzip BX-CSV-Dump.zip -d Book-Crossing
            python run.py --dataset ml-100k \ 
            --input_path ml-100k --output_path output_data/ml-100k \
            --convert_inter --convert_item --convert_user
            python run.py --dataset ml-1m \ 
            --input_path ml-1m --output_path output_data/ml-1m \
            --convert_inter --convert_item --convert_user
            python run.py --dataset book-crossing \
            --input_path Book-Crossing --output_path output_data/book-crossing \
            --convert_inter --convert_user --convert_item
   ```
3. Train Base RS models.
   ```bash
   python main.py --recsys_model <name of RS model> --dataset_name <name of RS model>
   ```
For runnnig run_llm file you would require API keys for GPT and Mixtral.
4. Evaluate LLM.
  ```bash
   python run_llm.py --recsys_model <name of RS model> --dataset_name <name of RS model> --language_model <name of LLM with version>
   ```  


