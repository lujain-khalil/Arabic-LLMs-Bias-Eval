import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import json
from utils import PALLETE, SUPPORTED_MODELS

results_dir = f"results/model_comparisions/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

eps_dir = os.path.join(results_dir, "eps")
if not os.path.exists(eps_dir):
    os.makedirs(eps_dir)

MODEL_COUNT = len(SUPPORTED_MODELS) # Assuming that results have been generated for all models, split equally between monolingual and multilingual models
FIG_SIZE = (12, 6)

def bar_plot(data, score, y_lim=None):
    flat_data = {model: values["Score"] for model, values in data.items()}
    df = pd.DataFrame({'Model': flat_data.keys(), 'Score': flat_data.values()})
    
    plt.figure(figsize=FIG_SIZE)
    ax = sns.barplot(x='Model', y='Score', data=df, hue='Model')
    
    if y_lim is not None:
        headroom = (y_lim[1] - y_lim[0]) * 0.05
        ax.set_ylim(y_lim[0] - headroom, y_lim[1] + headroom)

    ax.set_title(f"{score} for Cultural Terms Across Models", fontsize=20)
    ax.set_ylabel(f"{score} Score", fontsize=14)
    ax.set_xlabel("", fontsize=1)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    plt.setp(ax.get_xticklabels(), rotation=45)

    monolingual_count = MODEL_COUNT/2
    ax.axvline(x=monolingual_count - 0.5, color='grey', linestyle='--', linewidth=1)

    # Label the halves of the plot
    ax.text(monolingual_count // 2, ax.get_ylim()[1] * 0.95, "Monolingual", ha='center', va='center', fontsize=14, color='grey')
    ax.text(monolingual_count + (MODEL_COUNT - monolingual_count) // 2, ax.get_ylim()[1] * 0.95, "Multilingual", ha='center', va='center', fontsize=14, color='grey')

    plt.tight_layout()
    plt.savefig(f"{results_dir}{score.lower()}_scores.png")
    eps_path = os.path.join(eps_dir, f"{score.lower()}_scores.eps")
    plt.savefig(eps_path, format='eps')


def grouped_bar_plot(data, target, y_lim=None):
    tidy_data = []
    for model, cultures in data.items():
        for culture, score in cultures.items():
            tidy_data.append({
                "Model": model,
                "SAME": score,
                "Culture": culture
            })
    
    plt.figure(figsize=FIG_SIZE)
    barplot = sns.barplot(data=pd.DataFrame(tidy_data), x='Model', y='SAME', hue='Culture', palette=PALLETE)

    if y_lim is not None:
        headroom = (y_lim[1] - y_lim[0]) * 0.05
        plt.ylim(y_lim[0], y_lim[1] + headroom)

    for container in barplot.containers:
        barplot.bar_label(container, fmt='%.3f', padding=3)
    plt.setp(barplot.get_xticklabels(), rotation=45)

    leg = barplot.get_legend()
    leg.set_title("Culture")

    plt.title(f"SAME for Cultural {target} Across Models", fontsize=20)
    plt.ylabel("SAME Score", fontsize=14)
    plt.xlabel("", fontsize=1)
    plt.legend(title='Culture', fontsize=12, title_fontsize=14)
    
    monolingual_count = MODEL_COUNT/2
    barplot.axvline(x=monolingual_count - 0.5, color='grey', linestyle='--', linewidth=1)

    # Label the halves of the plot
    barplot.text(monolingual_count // 2, barplot.get_ylim()[1] * 0.95, "Monolingual", ha='center', va='center', fontsize=14, color='grey')
    barplot.text(monolingual_count + (MODEL_COUNT - monolingual_count) // 2, barplot.get_ylim()[1] * 0.95, "Multilingual", ha='center', va='center', fontsize=14, color='grey')

    plt.tight_layout()
    plt.savefig(f"{results_dir}same_{target.lower()}.png")
    eps_path = os.path.join(eps_dir, f"same_{target.lower()}.eps")
    plt.savefig(eps_path, format='eps')


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

weat_scores = [v["Score"] for v in combined_results["WEAT"].values()]
seat_scores = [v["Score"] for v in combined_results["SEAT"].values()]
global_y_min = min(min(weat_scores), min(seat_scores))
global_y_max = max(max(weat_scores), max(seat_scores))

same_terms_scores = [score for model in combined_results["SAME-terms"].values() for score in model.values()]
same_sentences_scores = [score for model in combined_results["SAME-sentences"].values() for score in model.values()]
global_same_y_min = min(min(same_terms_scores), min(same_sentences_scores))
global_same_y_max = max(max(same_terms_scores), max(same_sentences_scores))

# 1. WEAT for cultural terms
bar_plot(combined_results["WEAT"], "WEAT", y_lim=(global_y_min, global_y_max))

# 2. SEAT for cultural sentences
bar_plot(combined_results["SEAT"], "SEAT", y_lim=(global_y_min, global_y_max))

# 3. SAME for cultural terms
grouped_bar_plot(combined_results["SAME-terms"], "Terms", y_lim=(global_same_y_min, global_same_y_max))

# 4. SAME for cultural sentences
grouped_bar_plot(combined_results["SAME-sentences"], "Sentences", y_lim=(global_same_y_min, global_same_y_max))