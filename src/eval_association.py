import pandas as pd
import argparse
from utils import LANGUAGE, PURPLE, PALLETE, ENTITY_PALLETE, compute_seat_weat, compute_same, normalize_embedding
import os 
import json
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description="Evaluate embeddings using a specified multilingual masked LLM.")
parser.add_argument('model_name', type=str, help="Name of the model to use (e.g., 'xlm-roberta-base', 'mbert', 'gigabert')")

args = parser.parse_args()
MODEL_NAME = args.model_name

lang_type = "monolingual" if (MODEL_NAME in LANGUAGE["monolingual"]) else "multilingual"
results_dir = f"results/{lang_type}/{MODEL_NAME}/association/"

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

eps_dir = f"results/{lang_type}/{MODEL_NAME}/association/eps/"
if not os.path.exists(eps_dir):
    os.makedirs(eps_dir)

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

def bar_plot(data, score, y_lim=None):
    df = pd.DataFrame({'Category': data.keys(), 'Score': data.values()})
    
    plt.figure(figsize=(15, 6))
    temp_pallete = ENTITY_PALLETE
    temp_pallete[f'Total {score}'] = PURPLE
    ax = sns.barplot(x='Category', y='Score', data=df, hue='Category', palette=temp_pallete)
    
    if y_lim is not None:
        headroom = (y_lim[1] - y_lim[0]) * 0.05
        ax.set_ylim(y_lim[0] - headroom, y_lim[1] + headroom)

    ax.set_title(f"{score} for Cultural Terms ({MODEL_NAME})", fontsize=16)
    ax.set_ylabel(f"{score} Score", fontsize=14)
    ax.set_xlabel(f"Entity Type", fontsize=14)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}{score.lower()}_scores.png")
    eps_path = os.path.join(eps_dir, f"{score.lower()}_scores.eps")
    plt.savefig(eps_path, format='eps')

def grouped_bar_plot(data, target, y_lim=None):  
    tidy_data = []
    for term_category, cultures in data.items():
        for culture, score in cultures.items():
            tidy_data.append({
                "Entity": 'Total SAME' if term_category == f'SAME-{target.lower()}' else term_category.replace("SAME for ", "").replace(" terms", ""),
                "SAME": score,
                "Culture": culture
            })

    plt.figure(figsize=(15, 6))
    barplot = sns.barplot(data=pd.DataFrame(tidy_data), x='Entity', y='SAME', hue='Culture', palette=PALLETE)

    if y_lim is not None:
        headroom = (y_lim[1] - y_lim[0]) * 0.05
        plt.ylim(y_lim[0], y_lim[1] + headroom)
    
    for container in barplot.containers:
        barplot.bar_label(container, fmt='%.3f', padding=3)

    plt.title(f"SAME for Cultural {target} ({MODEL_NAME})", fontsize=16)
    plt.ylabel("SAME Score", fontsize=14)
    plt.xlabel(f"Entity Type", fontsize=14)

    plt.legend(title="Culture", fontsize=12, title_fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{results_dir}same_{target.lower()}.png")
    eps_path = os.path.join(eps_dir, f"same_{target.lower()}.eps")
    plt.savefig(eps_path, format='eps')

print(f"Genrating plots...")

# 1. WEAT for cultural terms
weat_data = {
    entity: combined_results[f"WEAT for {entity}"]["Score"]
    for entity in culture_embeddings['entity'].unique()}
weat_data['Total WEAT'] = combined_results["WEAT"]["Score"]

# 2. SEAT for cultural sentences
seat_data = {
    entity: combined_results[f"SEAT for {entity}"]["Score"]
    for entity in sentence_embeddings['entity'].unique()}
seat_data['Total SEAT'] = combined_results["SEAT"]["Score"]

# 3. SAME for cultural terms
same_terms_data = {
    entity: combined_results[f"SAME for {entity} terms"]
    for entity in culture_embeddings['entity'].unique()}
same_terms_data['SAME-terms'] = combined_results["SAME-terms"]

# 4. SAME for cultural sentences
same_sentences_data = {
    entity: combined_results[f"SAME for {entity} sentences"]
    for entity in sentence_embeddings['entity'].unique()}
same_sentences_data['SAME-sentences'] = combined_results["SAME-sentences"]

# Compute global y-limits for WEAT and SEAT plots
weat_values = list(weat_data.values())
seat_values = list(seat_data.values())
global_weat_seat_min = min(min(weat_values), min(seat_values))
global_weat_seat_max = max(max(weat_values), max(seat_values))

# Compute global y-limits for SAME plots
same_terms_values = [score for subdict in same_terms_data.values() for score in subdict.values()]
same_sentences_values = [score for subdict in same_sentences_data.values() for score in subdict.values()]
global_same_min = min(min(same_terms_values), min(same_sentences_values))
global_same_max = max(max(same_terms_values), max(same_sentences_values))

bar_plot(weat_data, "WEAT", y_lim=(global_weat_seat_min, global_weat_seat_max))
bar_plot(seat_data, "SEAT", y_lim=(global_weat_seat_min, global_weat_seat_max))

grouped_bar_plot(same_terms_data, "Terms", y_lim=(global_same_min, global_same_max))
grouped_bar_plot(same_sentences_data, "Sentences", y_lim=(global_same_min, global_same_max))

print(f"WEAT, SEAT, and SAME results saved in '{results_dir}asssociation_metrics_results.json'")