import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os 
import argparse
from utils import SUPPORTED_MODELS, PALLETE

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

# Stack them together
all_embeddings = np.vstack((arab_embeddings, western_embeddings))
print(all_embeddings.shape)

# Labels (Arab = 0, Western = 1)
labels = ['Arab'] * len(arab_embeddings) + ['Western'] * len(western_embeddings)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_embeddings = tsne.fit_transform(all_embeddings)

# Create a DataFrame for plotting
tsne_df = pd.DataFrame(tsne_embeddings, columns=['t-SNE 1', 't-SNE 2'])
tsne_df['Culture'] = labels

# Plot t-SNE results
plt.figure(figsize=(8, 6))
sns.scatterplot(x='t-SNE 1', y='t-SNE 2', hue='Culture', data=tsne_df, palette=PALLETE, s=100)
plt.title('t-SNE Visualization: Arab vs Western Terms')
plt.tight_layout()
plt.savefig(f"{results_dir}tsne_plot.png")