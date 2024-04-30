import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
def plot_sparsity_vs_auc(sorted_sparsity_values, sorted_auc_scores):
    user_ids = sorted_auc_scores.keys()

    # Get the corresponding AUC scores and sparsity scores
    auc_scores = [sorted_auc_scores[user_id] for user_id in user_ids]
    sparsity_scores = [sorted_sparsity_values[user_id] for user_id in user_ids]

    # Create the DataFrame
    df = pd.DataFrame({
        'user_id': list(user_ids),
        'auc_score': auc_scores,
        'sparsity': sparsity_scores})

    plt.figure(figsize=(10, 6), dpi=200)
    sns.set_style("whitegrid")
    df['distance_from_y_axis'] = abs(df['sparsity'])
    scatter = sns.scatterplot(data=df, x='sparsity', y='auc_score', hue='auc_score', palette='viridis', marker='.', size='distance_from_y_axis', sizes=(450, 70), edgecolor='black', linewidth=0.09)
    scatter.legend_.remove()

    #plt.title('Sparsity values vs AUC scores for each user', fontsize=16)
    plt.xlabel('Sparsity value', fontsize=19)
    plt.ylabel('AUC score', fontsize=19)
    plt.tick_params(axis='both', which='major', labelsize=19)

    average_sparsity = df['sparsity'].mean()
    filtered_df = df[(df['auc_score'] < 0.4) & (df['sparsity'] > (average_sparsity-0.5))]

    plt.show()
    return filtered_df
