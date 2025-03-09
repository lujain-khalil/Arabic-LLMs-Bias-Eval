import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from utils import LANGUAGE, GREEN, PURPLE, PALLETE, compute_norm, check_normality, perform_statistical_test
import os 
import json

plt.rcParams.update({'xtick.labelsize': 12, 'ytick.labelsize': 12})

parser = argparse.ArgumentParser(description="Evaluate embeddings using a specified multilingual masked LLM.")
parser.add_argument('model_name', type=str, help="Name of the model to use (e.g., 'xlm-roberta-base', 'mbert', 'gigabert')")

args = parser.parse_args()
MODEL_NAME = args.model_name

lang_type = "monolingual" if (MODEL_NAME in LANGUAGE["monolingual"]) else "multilingual"
results_dir = f"results/{lang_type}/{MODEL_NAME}/norms/"

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

eps_dir = f"results/{lang_type}/{MODEL_NAME}/norms/eps/"
if not os.path.exists(eps_dir):
    os.makedirs(eps_dir)

# Load the generated embeddings for cultural terms
embeddings_df = pd.read_pickle(f"embeddings/{MODEL_NAME}/culture_term_embeddings.pkl")

# Separate embeddings based on culture (Arab vs. Western)
arab_embeddings = embeddings_df[embeddings_df['culture'] == 'Arab']['embedding'].values
western_embeddings = embeddings_df[embeddings_df['culture'] == 'Western']['embedding'].values

# Compute norms for Arab and Western embeddings
arab_norms = np.array([compute_norm(embedding) for embedding in arab_embeddings])
western_norms = np.array([compute_norm(embedding) for embedding in western_embeddings])

# Calculate the mean and standard deviation of the norms for each group
arab_mean_norm = np.mean(arab_norms)
western_mean_norm = np.mean(western_norms)
arab_std_norm = np.std(arab_norms)
western_std_norm = np.std(western_norms)

mean_std_values = {
    'Arab': {'mean': float(arab_mean_norm), 'std': float(arab_std_norm)},
    'Western': {'mean': float(western_mean_norm), 'std': float(western_std_norm)}
}

result_data = {'mean_std_values': mean_std_values}

# Prepare data for boxplot, violin plot, and histogram
norms_df = pd.DataFrame({
    'Culture': ['Arab'] * len(arab_norms) + ['Western'] * len(western_norms),
    'Norm': np.concatenate([arab_norms, western_norms])
})

print('Generating plots...')
FIG_SIZE = (6,5)

# Boxplot
plt.figure(figsize=FIG_SIZE)
sns.boxplot(data=norms_df, x='Culture', y='Norm', hue='Culture', palette=PALLETE)
plt.title(f'Box Plot of Norms ({MODEL_NAME})', fontsize=20)
plt.xlabel('', fontsize=1)
plt.ylabel('Embedding Norm', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(f"{results_dir}boxplot_norms.png")
eps_path = os.path.join(eps_dir, "boxplot_norms.eps")
plt.savefig(eps_path, format='eps')
plt.close()

# Violin Plot
plt.figure(figsize=FIG_SIZE)
sns.violinplot(data=norms_df, x='Culture', y='Norm', hue='Culture', palette=PALLETE)
plt.title(f'Violin Plot of Norms ({MODEL_NAME})', fontsize=20)
plt.xlabel('', fontsize=1)
plt.ylabel('Embedding Norm', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(f"{results_dir}violin_plot_norms.png")
eps_path = os.path.join(eps_dir, "violin_plot_norms.eps")
plt.savefig(eps_path, format='eps')
plt.close()

# Histogram with KDE
plt.figure(figsize=FIG_SIZE)
sns.histplot(arab_norms, kde=True, color=GREEN, label='Arab', stat='density', bins=20, alpha=0.5)
sns.histplot(western_norms, kde=True, color=PURPLE, label='Western', stat='density', bins=20, alpha=0.5)
plt.title(f'Histogram of Norms ({MODEL_NAME})', fontsize=20)
plt.xlabel('', fontsize=1)
plt.ylabel('Density', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{results_dir}histogram_kde_norms.png")
eps_path = os.path.join(eps_dir, "histogram_kde_norms.eps")
plt.savefig(eps_path, format='eps')
plt.close()

# Comparision of cultural entities
culture_entity_df = embeddings_df.copy()
culture_entity_df['Norm'] = culture_entity_df['embedding'].apply(compute_norm)

# Calculate mean and std for each culture-entity
culture_entity_stats = culture_entity_df.groupby(['culture', 'entity'])['Norm'].agg(['mean', 'std']).reset_index()
result_data['culture_entity_stats'] = culture_entity_stats.to_dict(orient='records')

with open(f"{results_dir}mean_std_values.json", 'w') as json_file:
    json.dump(result_data, json_file, indent=4)

print(f"All visualizations for {MODEL_NAME} have been saved in '{results_dir}' directory.")

# Check normality for Arab and Western norms
arab_normality = check_normality(arab_norms)
western_normality = check_normality(western_norms)

# Perform statistical test based on normality results
statistical_results = perform_statistical_test(arab_norms, western_norms, arab_normality, western_normality)

# Combine all results
results = {
    'normality_tests': {
        'Arab': arab_normality,
        'Western': western_normality
    },
    'statistical_tests': statistical_results
}

with open(f"{results_dir}statistical_tests.json", 'w') as json_file:
    json.dump(results, json_file, indent=4)

print(f"Results saved in '{results_dir}statistical_tests.json'")