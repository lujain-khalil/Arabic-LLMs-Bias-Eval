import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import json
from utils import PALLETE, WEAT_SEAT_PALLETE, SUPPORTED_MODELS, grouped_barplot

results_dir = f"results/model_comparisions/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

eps_dir = os.path.join(results_dir, "eps")
if not os.path.exists(eps_dir):
    os.makedirs(eps_dir)

MODEL_COUNT = len(SUPPORTED_MODELS)

combined_results = {
    "WEAT": {},
    "SEAT": {},
    "SAME-terms": {},
    "SAME-sentences": {}
}

root_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

for lang_dir in ['monolingual', 'multilingual']:
    lang_dir_path = os.path.join(root_dir, lang_dir)
    
    for model_dir in os.listdir(lang_dir_path):
        model_dir_path = os.path.join(lang_dir_path, model_dir, 'association')
        json_file_path = os.path.join(model_dir_path, 'asssociation_metrics_results.json')

        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                data = json.load(f)

            for metric in combined_results.keys():
                if metric in ['SAME-terms', 'SAME-sentences']:
                    combined_results[metric][model_dir] = {}
                    for sub_metric, score in data[metric].items():
                        combined_results[metric][model_dir][sub_metric] = float(score)
                else:
                    combined_results[metric][model_dir] = {
                        "Score": float(data[metric]["Score"])
                    }

print(f"Saving results...")
with open(f"{results_dir}asssociation_metrics_model_comparision.json", 'w') as f:
    json.dump(combined_results, f, indent=4)

print(f"Generating plots...")

models = combined_results["WEAT"].keys()
model_names = [("XLM-Base" if m == "XLM-RoBERTa-Base" 
                else ("XLM-Large" if m == "XLM-RoBERTa-Large" else m))
                for m in models]

weat_scores = [v["Score"] for v in combined_results["WEAT"].values()]
seat_scores = [v["Score"] for v in combined_results["SEAT"].values()]
global_y_min = min(min(weat_scores), min(seat_scores))
global_y_max = max(max(weat_scores), max(seat_scores))

seat_weat_data = {
    "WEAT": weat_scores, 
    "SEAT": seat_scores
}

same_terms_scores = [score for model in combined_results["SAME-terms"].values() for score in model.values()]
same_sentences_scores = [score for model in combined_results["SAME-sentences"].values() for score in model.values()]
global_same_y_min = min(min(same_terms_scores), min(same_sentences_scores))
global_same_y_max = max(max(same_terms_scores), max(same_sentences_scores))

same_terms_data = {
    "Arab": [combined_results["SAME-terms"][model]["Arab"] for model in models],
    "Western": [combined_results["SAME-terms"][model]["Western"] for model in models]
}

same_sentences_data = {
    "Arab": [combined_results["SAME-sentences"][model]["Arab"] for model in models],
    "Western": [combined_results["SAME-sentences"][model]["Western"] for model in models]
}

grouped_barplot(
    group_names=model_names, 
    scenario_data=seat_weat_data, 
    title=f"WEAT and SEAT Scores Across Models", 
    ylabel="Score",
    hline=None,
    vline=True, 
    colors=WEAT_SEAT_PALLETE,
    ylim=[global_y_min, global_y_max],
    output_png=os.path.join(results_dir, "weat_seat_scores.png"),
    output_eps=os.path.join(eps_dir, "weat_seat_scores.eps")
)

grouped_barplot(
    group_names=model_names, 
    scenario_data=same_terms_data, 
    title=f"SAME for Cultural Terms Across Models", 
    ylabel="SAME Score",
    hline=None,
    vline=True, 
    colors=PALLETE,
    ylim=[global_same_y_min, global_same_y_max],
    output_png=os.path.join(results_dir, "same_terms.png"),
    output_eps=os.path.join(eps_dir, "same_terms.eps")
)

grouped_barplot(
    group_names=model_names, 
    scenario_data=same_sentences_data, 
    title=f"SAME for Cultural Sentences Across Models", 
    ylabel="SAME Score",
    hline=None,
    vline=True, 
    colors=PALLETE,
    ylim=[global_same_y_min, global_same_y_max],
    output_png=os.path.join(results_dir, "same_sentences.png"),
    output_eps=os.path.join(eps_dir, "same_sentences.eps")
)