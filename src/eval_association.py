import pandas as pd
import numpy as np
import argparse
from utils import SUPPORTED_MODELS, PURPLE, GREEN, PALLETE, ENTITY_PALLETE, compute_seat_weat, compute_same, normalize_embedding
import os 
import json
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description="Evaluate embeddings using a specified multilingual masked LLM.")
parser.add_argument('model_name', type=str, choices=SUPPORTED_MODELS.keys(), help="Name of the model to use (e.g., 'xlm-roberta-base', 'mbert', 'gigabert')")

args = parser.parse_args()
MODEL_NAME = args.model_name

results_dir = f"results/{MODEL_NAME}/association/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load the generated embeddings for cultural terms
print("Reading pickle files...")
culture_embeddings = pd.read_pickle(f"embeddings/{MODEL_NAME}/culture_term_embeddings.pkl")
sentence_embeddings = pd.read_pickle(f"embeddings/{MODEL_NAME}/context_sentence_embeddings.pkl")
sentiment_embeddings = pd.read_pickle(f"embeddings/{MODEL_NAME}/sentiment_term_embeddings.pkl")

print("Normalizing embeddings...")
culture_embeddings['embedding'] = culture_embeddings['embedding'].apply(normalize_embedding)
sentence_embeddings['embedding'] = sentence_embeddings['embedding'].apply(normalize_embedding)
sentiment_embeddings['embedding'] = sentiment_embeddings['embedding'].apply(normalize_embedding)

combined_results = {}

print("Computing WEAT...")
combined_results['WEAT'] = compute_seat_weat(culture_embeddings, sentiment_embeddings)

print("Computing WEAT for each entity...")
for entity in culture_embeddings['entity'].unique():
    entity_embeddings = culture_embeddings[culture_embeddings['entity'] == entity]
    entity_results = compute_seat_weat(entity_embeddings, sentiment_embeddings)

    combined_results[f'WEAT for {entity}'] = entity_results

print("Computing SEAT...")
combined_results['SEAT'] = compute_seat_weat(sentence_embeddings, sentiment_embeddings)

print("Computing SEAT for each entity...")
for entity in sentence_embeddings['entity'].unique():
    entity_embeddings = sentence_embeddings[sentence_embeddings['entity'] == entity]
    entity_results = compute_seat_weat(entity_embeddings, sentiment_embeddings)

    combined_results[f'SEAT for {entity}'] = entity_results

print('Computing SAME for terms...')
combined_results['SAME-terms'] = compute_same(culture_embeddings, sentiment_embeddings)

print('Computing SAME for sentences...')
combined_results['SAME-sentences'] = compute_same(sentence_embeddings, sentiment_embeddings)

print("Computing SAME for terms by entity...")
for entity in culture_embeddings['entity'].unique():
    entity_embeddings = culture_embeddings[culture_embeddings['entity'] == entity]
    entity_results = compute_same(entity_embeddings, sentiment_embeddings)

    combined_results[f'SAME for {entity} terms'] = entity_results

print("Computing SAME for sentences by entity...")
for entity in culture_embeddings['entity'].unique():
    entity_embeddings = culture_embeddings[culture_embeddings['entity'] == entity]
    entity_results = compute_same(entity_embeddings, sentiment_embeddings)

    combined_results[f'SAME for {entity} sentences'] = entity_results

print(f"Saving results...")
with open(f"{results_dir}asssociation_metrics_results.json", 'w') as f:
    json.dump(combined_results, f, indent=4)

# ------------ Generating plots ------------

import seaborn as sns
import pandas as pd

def bar_plot(data, score):
    df = pd.DataFrame({'Category': data.keys(), 'Score': data.values()})
    
    plt.figure(figsize=(15, 6))
    temp_pallete = ENTITY_PALLETE
    temp_pallete[score] = PURPLE
    ax = sns.barplot(x='Category', y='Score', data=df, hue='Category', palette=temp_pallete, legend=False)
    
    ax.set_title(f"{score} for Cultural Terms ({MODEL_NAME})")
    ax.set_ylabel(f"{score} Score")
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}{score.lower()}_scores.png")

def grouped_bar_plot(data, target):  
    tidy_data = []
    for term_category, cultures in data.items():
        for culture, score in cultures.items():
            tidy_data.append({
                "entity": 'SAME' if term_category == f'SAME-{target.lower()}' else term_category.replace("SAME for ", "").replace(" terms", ""),
                "SAME": score,
                "culture": culture
            })

    plt.figure(figsize=(15, 6))
    barplot = sns.barplot(data=pd.DataFrame(tidy_data), x='entity', y='SAME', hue='culture', palette=PALLETE)

    for container in barplot.containers:
        barplot.bar_label(container, fmt='%.3f', padding=3)

    plt.title(f"SAME for Cultural {target} ({MODEL_NAME})")
    plt.ylabel("SAME Score")
    plt.tight_layout()
    plt.savefig(f"{results_dir}same_{target.lower()}.png")

def association_scores_plot(target, arab_means, arab_stds, western_means, western_stds, target_name):
    target.append(target_name)
    
    data = pd.DataFrame({
        'Target': target * 2,
        'Mean': arab_means + western_means,
        'Std': arab_stds + western_stds,
        'Group': ['Arab'] * len(target) + ['Western'] * len(target)
    })

    plt.figure(figsize=(15, 6))
    ax = sns.barplot(x='Target', y='Mean', hue='Group', data=data, errorbar=None, palette=PALLETE)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
        
    ax.set_title(f"Mean and Standard Deviation of {score} Association Scores ({MODEL_NAME})")
    ax.set_ylabel("Association Score")
    ax.set_xlabel("Target")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}{score.lower()}_association_scores.png")

print(f"Genrating plots...")

# 1. WEAT for cultural terms
weat_data = {
    entity: combined_results[f"WEAT for {entity}"]["Score"]
    for entity in culture_embeddings['entity'].unique()
}
weat_data['WEAT'] = combined_results["WEAT"]["Score"]
bar_plot(weat_data, "WEAT")

# 2. SEAT for cultural sentences
seat_data = {
    entity: combined_results[f"SEAT for {entity}"]["Score"]
    for entity in sentence_embeddings['entity'].unique()
}
seat_data['SEAT'] = combined_results["SEAT"]["Score"]
bar_plot(seat_data, "SEAT")

# 3. SAME for cultural terms
same_terms_data = {
    entity: combined_results[f"SAME for {entity} terms"]
    for entity in culture_embeddings['entity'].unique()
}
same_terms_data['SAME-terms'] = combined_results["SAME-terms"]
grouped_bar_plot(same_terms_data, "Terms")

# 4. SAME for cultural sentences
same_sentences_data = {
    entity: combined_results[f"SAME for {entity} sentences"]
    for entity in sentence_embeddings['entity'].unique()
}
same_sentences_data['SAME-sentences'] = combined_results["SAME-sentences"]
grouped_bar_plot(same_sentences_data, "Sentences")

# 6. Association scores for SEAT and WEAT
for score in ['WEAT', 'SEAT']:
    entities = list(culture_embeddings['entity'].unique())

    arab_means = [combined_results[f"{score} for {entity}"]["Mean Arab (M)"] for entity in entities]
    arab_means.append(combined_results[score]["Mean Arab (M)"])

    arab_stds = [combined_results[f"{score} for {entity}"]["Std Arab (M)"] for entity in entities]
    arab_stds.append(combined_results[score]["Std Arab (M)"])

    western_means = [combined_results[f"{score} for {entity}"]["Mean Western (F)"] for entity in entities]
    western_means.append(combined_results[score]["Mean Western (F)"])

    western_stds = [combined_results[f"{score} for {entity}"]["Std Western (F)"] for entity in entities]
    western_stds.append(combined_results[score]["Std Western (F)"])

    association_scores_plot(entities, arab_means, arab_stds, western_means, western_stds, score)

print(f"WEAT, SEAT, and SAME results saved in '{results_dir}asssociation_metrics_results.json'")