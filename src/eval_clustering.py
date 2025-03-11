import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os 
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
from utils import LANGUAGE, PALLETE, ENTITY_PALLETE, CULTURE_ENTITY_PALLETE, compute_cluster_distances, map_labels


parser = argparse.ArgumentParser(description="Evaluate embeddings using a specified multilingual masked LLM.")
parser.add_argument('model_name', type=str, help="Name of the model to use (e.g., 'xlm-roberta-base', 'mbert', 'gigabert')")

args = parser.parse_args()
MODEL_NAME = args.model_name

lang_type = "monolingual" if (MODEL_NAME in LANGUAGE["monolingual"]) else "multilingual"
results_dir = f"results/{lang_type}/{MODEL_NAME}/clustering/"

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

eps_dir = f"results/{lang_type}/{MODEL_NAME}/clustering/eps/"
if not os.path.exists(eps_dir):
    os.makedirs(eps_dir)

embeddings_df = pd.read_pickle(f"embeddings/{MODEL_NAME}/culture_term_embeddings.pkl")

# Combine all embeddings for Arab and Western terms
arab_embeddings = embeddings_df[embeddings_df['culture'] == 'Arab']['embedding'].values
western_embeddings = embeddings_df[embeddings_df['culture'] == 'Western']['embedding'].values

arab_embeddings = np.array([embedding.flatten() for embedding in arab_embeddings])
western_embeddings = np.array([embedding.flatten() for embedding in western_embeddings])

all_embeddings = np.vstack((arab_embeddings, western_embeddings))

# Extract culture and entity labels
arab_labels = embeddings_df[embeddings_df['culture'] == 'Arab'][['culture', 'entity', 'term']].values
western_labels = embeddings_df[embeddings_df['culture'] == 'Western'][['culture', 'entity', 'term']].values

all_labels = np.vstack((arab_labels, western_labels))

# Apply t-SNE
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42)
tsne_embeddings = tsne.fit_transform(all_embeddings)

# Create a DataFrame for plotting
tsne_df = pd.DataFrame(tsne_embeddings, columns=['t-SNE 1', 't-SNE 2'])
tsne_df['Culture'] = [label[0] for label in all_labels] 
tsne_df['Entity'] = [label[1] for label in all_labels] 
tsne_df['Term'] = [label[2] for label in all_labels] 
tsne_df['Culture-Entity'] = [f"{label[0]}-{label[1]}" for label in all_labels]

# Apply k-means 
print("Running K-Means Clustering by Culture...")
kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(all_embeddings)

# Silhouette score, Intra-Cluster and Inter-Cluster Distances
silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
intra_culture, inter_culture = compute_cluster_distances(all_embeddings, tsne_df['Culture'].values, 'Culture')
intra_culture_entity, inter_culture_entity = compute_cluster_distances(all_embeddings, tsne_df['Culture-Entity'].values, 'Culture-Entity')

# Plots
print(f"Generating plots...")
FIG_SIZE=(6,5)

def scatter_plot(df, column, save_path, eps_dir, palette = 'flare'):
    plt.figure(figsize=FIG_SIZE)
    sns.scatterplot(
        x='t-SNE 1',
        y='t-SNE 2',
        hue=column,
        data=df,
        palette=palette,
        s=100,
        legend=False
    )
    cluster_centroids = df.groupby(column)[['t-SNE 1', 't-SNE 2']].mean()
    cluster_label_size = {
        'Culture': 16,
        'Entity': 14,
        'Culture-Entity': 12
    }
    for cluster_name, (x, y) in cluster_centroids.iterrows():
        plt.text(
            x, y, str(cluster_name),
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=cluster_label_size[column],
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
        )
    plt.title(f't-SNE by {column} ({MODEL_NAME})', fontsize=18)
    plt.xlabel("", fontsize=1)
    plt.ylabel("", fontsize=1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path)
    eps_filename = os.path.join(eps_dir, os.path.basename(save_path).replace('.png', '.eps'))
    plt.savefig(eps_filename, format='eps')
    plt.savefig(save_path)

scatter_plot(tsne_df, 'Culture', f"{results_dir}tsne_plot_culture.png", eps_dir, PALLETE)
scatter_plot(tsne_df, 'Entity', f"{results_dir}tsne_plot_entity.png", eps_dir, ENTITY_PALLETE)
scatter_plot(tsne_df, 'Culture-Entity', f"{results_dir}tsne_plot_culture_entity.png", eps_dir, CULTURE_ENTITY_PALLETE)

# Save results
print(f"Saving results...")
results = {
    "Silhouette Score (K-Means)": float(silhouette_avg),
    "Intra-Cluster Distance (Culture)": float(intra_culture),
    "Inter-Cluster Distance (Culture)": float(inter_culture),
    "Intra-Cluster Distance (Culture-Entity)": float(intra_culture_entity),
    "Inter-Cluster Distance (Culture-Entity)": float(inter_culture_entity),
}

with open(f"{results_dir}clustering_results.json", 'w') as json_file:
    json.dump(results, json_file, indent=4)

print(f"Clustering results saved in '{results_dir}clustering_results.json'")