import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os 
import argparse
import json
from utils import SUPPORTED_MODELS, PALLETE, compute_cluster_distances, scatter_plot, map_labels

parser = argparse.ArgumentParser(description="Evaluate embeddings using a specified multilingual masked LLM.")
parser.add_argument('model_name', type=str, choices=SUPPORTED_MODELS.keys(), help="Name of the model to use (e.g., 'xlm-roberta-base', 'mbert', 'gigabert')")

args = parser.parse_args()
MODEL_NAME = args.model_name

results_dir = f"results/{MODEL_NAME}/clustering/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

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
tsne_df['KMeans_Cluster'] = map_labels(cluster_labels, tsne_df)

# Silhouette score, Intra-Cluster and Inter-Cluster Distances
silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
intra_culture, inter_culture = compute_cluster_distances(all_embeddings, tsne_df['Culture'].values, 'Culture')
intra_culture_entity, inter_culture_entity = compute_cluster_distances(all_embeddings, tsne_df['Culture-Entity'].values, 'Culture-Entity')

# Plots
print(f"Plotting...")
scatter_plot(tsne_df, 'Culture', f't-SNE Visualization: Grouped by Culture ({MODEL_NAME})', f"{results_dir}tsne_plot_culture.png", PALLETE)
scatter_plot(tsne_df, 'Entity', f't-SNE Visualization: Grouped by Entity ({MODEL_NAME})', f"{results_dir}tsne_plot_entity.png")
scatter_plot(tsne_df, 'Culture-Entity', f't-SNE Visualization: Grouped by Culture-Entity ({MODEL_NAME})', f"{results_dir}tsne_plot_culture_entity.png")
scatter_plot(tsne_df, 'KMeans_Cluster', f'K-Means Clustering Results ({MODEL_NAME})', f"{results_dir}kmeans_clusters.png", PALLETE)

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