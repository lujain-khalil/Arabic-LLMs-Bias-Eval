import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import json
from utils import PALLETE, SUPPORTED_MODELS

results_dir = f"results/model_comparisions/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

MODEL_COUNT = len(SUPPORTED_MODELS) # Assuming that results have been generated for all models, split equally between monolingual and multilingual models

def bar_plot(data, score):
    flat_data = {model: values["Score"] for model, values in data.items()}
    df = pd.DataFrame({'Model': flat_data.keys(), 'Score': flat_data.values()})
    
    plt.figure(figsize=(15, 6))
    ax = sns.barplot(x='Model', y='Score', data=df, hue='Model', legend=False)
    
    ax.set_title(f"{score} for Cultural Terms Across Models")
    ax.set_ylabel(f"{score} Score")
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    
    monolingual_count = MODEL_COUNT/2
    ax.axvline(x=monolingual_count - 0.5, color='grey', linestyle='--', linewidth=1)

    # Label the halves of the plot
    ax.text(monolingual_count // 2, ax.get_ylim()[1] * 0.95, "Monolingual", ha='center', va='center', fontsize=14, color='grey')
    ax.text(monolingual_count + (MODEL_COUNT - monolingual_count) // 2, ax.get_ylim()[1] * 0.95, "Multilingual", ha='center', va='center', fontsize=14, color='grey')

    plt.tight_layout()
    plt.savefig(f"{results_dir}{score.lower()}_scores.png")


def grouped_bar_plot(data, target):
    tidy_data = []
    for model, cultures in data.items():
        for culture, score in cultures.items():
            tidy_data.append({
                "model": model,
                "SAME": score,
                "culture": culture
            })

    plt.figure(figsize=(15, 6))
    barplot = sns.barplot(data=pd.DataFrame(tidy_data), x='model', y='SAME', hue='culture', palette=PALLETE)

    for container in barplot.containers:
        barplot.bar_label(container, fmt='%.3f', padding=3)

    plt.title(f"SAME for Cultural {target} Across Models")
    plt.ylabel("SAME Score")

    monolingual_count = MODEL_COUNT/2
    barplot.axvline(x=monolingual_count - 0.5, color='grey', linestyle='--', linewidth=1)

    # Label the halves of the plot
    barplot.text(monolingual_count // 2, barplot.get_ylim()[1] * 0.95, "Monolingual", ha='center', va='center', fontsize=14, color='grey')
    barplot.text(monolingual_count + (MODEL_COUNT - monolingual_count) // 2, barplot.get_ylim()[1] * 0.95, "Multilingual", ha='center', va='center', fontsize=14, color='grey')

    plt.tight_layout()
    plt.savefig(f"{results_dir}same_{target.lower()}.png")

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

            for metric in ['WEAT', 'SEAT', 'SAME-terms', 'SAME-sentences']:
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

# 1. WEAT for cultural terms
bar_plot(combined_results["WEAT"], "WEAT")

# 2. SEAT for cultural sentences
bar_plot(combined_results["SEAT"], "SEAT")

# 3. SAME for cultural terms
grouped_bar_plot(combined_results["SAME-terms"], "Terms")

# 4. SAME for cultural sentences
grouped_bar_plot(combined_results["SAME-sentences"], "Sentences")