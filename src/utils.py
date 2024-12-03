from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

SUPPORTED_MODELS = {
    "xlm-roberta-base": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
    "gigabert": "nlpaueb/legal-bert-base-uncased",
}

GREEN = '#90c926'  
PURPLE = '#5f26c9'
PALLETE = {'Arab': GREEN, 'Western': PURPLE}

# Embedding Function
def get_embedding(sentence, model_name):
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Invalid model name. Supported models are: {', '.join(SUPPORTED_MODELS.keys())}")

    # Load tokenizer and model
    model_path = SUPPORTED_MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    # Tokenize input sentence
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def compute_norm(embedding):
    return np.linalg.norm(embedding)

# Helper function to check normality using multiple tests
def check_normality(data):
    normality_results = {}

    # Shapiro-Wilk Test
    shapiro_stat, shapiro_pvalue = stats.shapiro(data)
    shapiro_stat, shapiro_pvalue = float(shapiro_stat), float(shapiro_pvalue)
    normality_results['shapiro_wilk'] = {
        'statistic': (shapiro_stat),
        'p_value': shapiro_pvalue,
        'normal': shapiro_pvalue > 0.05  # Normal if p-value > 0.05
    }

    # Kolmogorov-Smirnov Test
    ks_stat, ks_pvalue = stats.kstest(data, 'norm')
    ks_stat, ks_pvalue = float(ks_stat), float(ks_pvalue)
    normality_results['kolmogorov_smirnov'] = {
        'statistic': ks_stat,
        'p_value': ks_pvalue,
        'normal': ks_pvalue > 0.05  # Normal if p-value > 0.05
    }

    return normality_results

# Function to perform t-test or Mann-Whitney U test based on normality results
def perform_statistical_test(arab_norms, western_norms, normality_results):
    statistical_test_results = {}

    # If both are normal, use a t-test
    if normality_results['shapiro_wilk']['normal'] and normality_results['kolmogorov_smirnov']['normal']:
        t_stat, t_pvalue = stats.ttest_ind(arab_norms, western_norms)
        t_stat, t_pvalue = float(t_stat), float(t_pvalue)
        statistical_test_results['t_test'] = {
            'statistic': t_stat,
            'p_value': t_pvalue,
            'significant': t_pvalue < 0.05  # Significant if p-value < 0.05
        }
    else:
        # If either of the distributions is not normal, use Mann-Whitney U test
        u_stat, u_pvalue = stats.mannwhitneyu(arab_norms, western_norms)
        u_stat, u_pvalue = float(u_stat), float(u_pvalue)
        statistical_test_results['mann_whitney_u'] = {
            'statistic': u_stat,
            'p_value': u_pvalue,
            'significant': u_pvalue < 0.05  # Significant if p-value < 0.05
        }

    return statistical_test_results

def compute_cluster_distances(embeddings, labels, group_label):
    intra_cluster_distances = []
    inter_cluster_distances = []

    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = embeddings[labels == label]
        intra_distance = np.mean(cdist(cluster_points, cluster_points))
        intra_cluster_distances.append(intra_distance)

        for other_label in unique_labels:
            if label != other_label:
                other_points = embeddings[labels == other_label]
                inter_distance = np.mean(cdist(cluster_points, other_points))
                inter_cluster_distances.append(inter_distance)

    return np.mean(intra_cluster_distances), np.mean(inter_cluster_distances)

def scatter_plot(df, column, title, save_path, palette = 'flare'):
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='t-SNE 1',
        y='t-SNE 2',
        hue=column,
        data=df,
        palette=palette,
        s=100,
    )
    # for _, row in df.iterrows():
    #     ax.annotate(
    #         row['Term'],
    #         (row['t-SNE 1'], row['t-SNE 2']),
    #         textcoords="offset points",
    #         xytext=(5, 5),
    #         ha='center',
    #         fontsize=8
    #     )
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(save_path)

def map_labels(labels, df):
    cluster_mapping = {}
    for cluster in np.unique(labels):
        indices = np.where(labels == cluster)[0]
        actual_cultures = df.iloc[indices]['Culture']

        majority_label = actual_cultures.mode()[0]
        cluster_mapping[cluster] = majority_label

    return np.array([cluster_mapping[label] for label in labels])
