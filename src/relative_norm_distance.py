import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from utils import SUPPORTED_MODELS
import os 

# Set up command-line argument parser
parser = argparse.ArgumentParser(description="Evaluate embeddings using a specified multilingual masked LLM.")
parser.add_argument('model_name', type=str, choices=SUPPORTED_MODELS.keys(),
                    help="Name of the model to use (e.g., 'xlm-roberta-base', 'mbert', 'gigabert')")

# Parse command-line arguments
args = parser.parse_args()
MODEL_NAME = args.model_name

results_dir = f"results/{MODEL_NAME}/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# Load the embeddings for the context sentences, cultural terms, and sentiment words
sentence_embeddings_df = pd.read_pickle(f"embeddings/{MODEL_NAME}/sentence_embeddings.pkl")
culture_term_embeddings_df = pd.read_pickle(f"embeddings/{MODEL_NAME}/culture_term_embeddings.pkl")
sentiment_term_embeddings_df = pd.read_pickle(f"embeddings/{MODEL_NAME}/sentiment_term_embeddings.pkl")

# Function to compute the Euclidean norm (magnitude) of embeddings
def compute_norm(embedding):
    return np.linalg.norm(embedding)

# Function to calculate the average norm and standard deviation for each group (Arab vs. Western)
def calculate_norms(df, group_col, embedding_col):
    arab_norms = []
    western_norms = []
    
    for _, row in df.iterrows():
        embedding = row[embedding_col]
        norm = compute_norm(embedding)
        
        if row[group_col] == 'arab':
            arab_norms.append(norm)
        elif row[group_col] == 'western':
            western_norms.append(norm)
    
    arab_mean = np.mean(arab_norms)
    western_mean = np.mean(western_norms)
    arab_std = np.std(arab_norms)
    western_std = np.std(western_norms)
    
    return arab_mean, arab_std, western_mean, western_std

# Compute the norms for culture terms
print("Computing norms for culture terms...")
arab_mean, arab_std, western_mean, western_std = calculate_norms(culture_term_embeddings_df, 'Culture', 'Embedding')

# Print the results
print(f"Arab Terms - Mean: {arab_mean:.3f}, Std: {arab_std:.3f}")
print(f"Western Terms - Mean: {western_mean:.3f}, Std: {western_std:.3f}")

# Bar chart visualization
print("Generating bar chart...")
categories = ['Arab Terms', 'Western Terms']
means = [arab_mean, western_mean]
stds = [arab_std, western_std]

fig, ax = plt.subplots()
ax.bar(categories, means, yerr=stds, capsize=5, color=['blue', 'orange'])
ax.set_ylabel('Mean Norm')
ax.set_title('Comparison of Mean Norms for Arab vs. Western Terms')

# Save the plot as a PNG file
print("Saving bar chart...")
plt.savefig(f"{results_dir}relative_norm_distance.png")
plt.show()

# Save the numerical results as a CSV file for reference
print("Saving numerical results...")
norm_results = {
    'Culture': ['Arab', 'Western'],
    'Mean Norm': [arab_mean, western_mean],
    'Std Dev': [arab_std, western_std]
}
norm_results_df = pd.DataFrame(norm_results)
norm_results_df.to_csv(f"{results_dir}relative_norm_distance_results.csv", index=False)

print("Relative norm distance evaluation completed and saved!")