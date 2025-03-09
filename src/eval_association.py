import pandas as pd
import argparse
from utils import LANGUAGE, PALLETE, WEAT_SEAT_PALLETE, compute_seat_weat, compute_same, normalize_embedding, grouped_barplot, process_same_data, process_seat_weat_data
import os 
import json

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
print(f"Genrating plots...")

# SAME for cultural terms
same_terms_data = {
    entity: combined_results[f"SAME for {entity} terms"]
    for entity in culture_embeddings['entity'].unique()}
same_terms_data['SAME-terms'] = combined_results["SAME-terms"]

# SAME for cultural sentences
same_sentences_data = {
    entity: combined_results[f"SAME for {entity} sentences"]
    for entity in sentence_embeddings['entity'].unique()}
same_sentences_data['SAME-sentences'] = combined_results["SAME-sentences"]

# Compute global y-limits for SAME plots
same_terms_values = [score for subdict in same_terms_data.values() for score in subdict.values()]
same_sentences_values = [score for subdict in same_sentences_data.values() for score in subdict.values()]
global_same_min = min(min(same_terms_values), min(same_sentences_values))
global_same_max = max(max(same_terms_values), max(same_sentences_values))

entities_seat_weat, seat_weat_data_processed, global_weat_seat_max, global_weat_seat_min = process_seat_weat_data(combined_results, 
                                                                                                                  culture_embeddings, 
                                                                                                                  sentence_embeddings)

entities_terms, same_terms_data_processed = process_same_data(same_terms_data, "terms")
entities_sentences, same_sentences_data_processed = process_same_data(same_sentences_data, "sentences")

grouped_barplot(
    group_names=entities_seat_weat,
    scenario_data=seat_weat_data_processed,
    title=f"WEAT and SEAT Scores by Entity ({MODEL_NAME})",
    ylabel="Score",
    hline=None,
    vline=False,
    colors=WEAT_SEAT_PALLETE,
    ylim=[global_weat_seat_min, global_weat_seat_max],
    output_png=os.path.join(results_dir, "weat_seat_scores.png"),
    output_eps=os.path.join(eps_dir, "weat_seat_scores.eps")
)

grouped_barplot(
    group_names=entities_terms,
    scenario_data=same_terms_data_processed,
    title=f"SAME for Cultural Terms ({MODEL_NAME})",
    ylabel="SAME Score",
    hline=None,
    vline=False,
    colors=PALLETE,
    ylim=[global_same_min, global_same_max],
    output_png=os.path.join(results_dir, "same_terms.png"),
    output_eps=os.path.join(eps_dir, "same_terms.eps")
)

grouped_barplot(
    group_names=entities_sentences,
    scenario_data=same_sentences_data_processed,
    title=f"SAME for Cultural Sentences ({MODEL_NAME})",
    ylabel="SAME Score",
    hline=None,
    vline=False,
    colors=PALLETE,
    ylim=[global_same_min, global_same_max],
    output_png=os.path.join(results_dir, "same_sentences.png"),
    output_eps=os.path.join(eps_dir, "same_sentences.eps")
)

print(f"WEAT, SEAT, and SAME results saved in '{results_dir}asssociation_metrics_results.json'")