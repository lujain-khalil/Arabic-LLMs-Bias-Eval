from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import cdist, cosine
import matplotlib.pyplot as plt

SUPPORTED_MODELS = {
    # Monolingual
    "AraBERT": "aubmindlab/bert-base-arabertv2",  
    "AraBERT-Large":"aubmindlab/bert-large-arabertv02",
    "ARBERT": "UBC-NLP/ARBERTv2", 
    "CAMeLBERT": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "MARBERT": "UBC-NLP/MARBERTv2",

    # Multiligual
    "BERT":"google-bert/bert-base-multilingual-uncased",
    "mBERT": "bert-base-multilingual-cased",
    "DistilBERT": "distilbert/distilbert-base-multilingual-cased",
    "XLM-RoBERTa-Base": "xlm-roberta-base",
    "XLM-RoBERTa-Large": "xlm-roberta-large"
}

LANGUAGE = {
    "multilingual": ["BERT", "mBERT", "DistilBERT", "XLM-RoBERTa-Base", "XLM-RoBERTa-Large"],
    "monolingual": ["AraBERT", "AraBERT-Large", "ARBERT", "CAMeLBERT", "MARBERT"]
}

GREEN = '#90c926'  
PURPLE = '#5f26c9'
PALLETE = {'Arab': GREEN, 'Western': PURPLE}
WEAT_SEAT_PALLETE = {'WEAT': "#8c564b", 'SEAT': "#e377c2"}

ENTITY_PALLETE = {
    "authors": "#1f77b4",          # Blue
    "beverage": "#ff7f0e",         # Orange
    "clothing-female": "#2ca02c",  # Green
    "clothing-male": "#d62728",    # Red
    "food": "#9467bd",             # Purple
    "location": "#8c564b",         # Brown
    "names-female": "#e377c2",     # Pink
    "names-male": "#7f7f7f",       # Gray
    "religious places": "#bcbd22", # Yellow-Green
    "sports clubs": "#17becf"      # Cyan
}

CULTURE_ENTITY_PALLETE = {
    "Arab-authors": "#1f77b4",          # Blue
    "Western-authors": "#154c70",       # Darker Blue
    "Arab-beverage": "#ff7f0e",         # Orange
    "Western-beverage": "#b25907",      # Darker Orange
    "Arab-clothing-female": "#2ca02c",  # Green
    "Western-clothing-female": "#196a19",# Darker Green
    "Arab-clothing-male": "#d62728",    # Red
    "Western-clothing-male": "#8b1b1c", # Darker Red
    "Arab-food": "#9467bd",             # Purple
    "Western-food": "#5e3d80",          # Darker Purple
    "Arab-location": "#8c564b",         # Brown
    "Western-location": "#5b392e",      # Darker Brown
    "Arab-names-female": "#e377c2",     # Pink
    "Western-names-female": "#924e7c",  # Darker Pink
    "Arab-names-male": "#7f7f7f",       # Gray
    "Western-names-male": "#4d4d4d",    # Darker Gray
    "Arab-religious places": "#bcbd22", # Yellow-Green
    "Western-religious places": "#7a7c16", # Darker Yellow-Green
    "Arab-sports clubs": "#17becf",     # Cyan
    "Western-sports clubs": "#0d7a86"   # Darker Cyan
}

BASE_ENTITY_FILES = {
        "authors": "authors.xlsx",
        "beverage": "beverage.xlsx",
        "clothing-female": "clothing-female.xlsx",
        "clothing-male": "clothing-male.xlsx",
        "food": "food.xlsx",
        "location": "locations.xlsx",
        "names-female": "names-female.xlsx",
        "names-male": "names-male.xlsx",
        "religious places": "religious-places.xlsx",
        "sports clubs": "sports-clubs.xlsx"
    }

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

def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError("Cannot normalize an embedding with zero norm.")
    return embedding / norm

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

def perform_statistical_test(arab_norms, western_norms, arab_normality, western_normality):
    statistical_test_results = {}
    is_normal = arab_normality['shapiro_wilk']['normal'] and arab_normality['kolmogorov_smirnov']['normal'] and western_normality['shapiro_wilk']['normal'] and western_normality['kolmogorov_smirnov']['normal']
    # If both are normal, use a t-test
    if is_normal:
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

def map_labels(labels, df):
    cluster_mapping = {}
    for cluster in np.unique(labels):
        indices = np.where(labels == cluster)[0]
        actual_cultures = df.iloc[indices]['Culture']

        majority_label = actual_cultures.mode()[0]
        cluster_mapping[cluster] = majority_label

    return np.array([cluster_mapping[label] for label in labels])

def compute_association(w, A, B):
    mean_cos_A = np.mean([1 - cosine(w, a) for a in A])
    mean_cos_B = np.mean([1 - cosine(w, b) for b in B])
    return mean_cos_A - mean_cos_B

def compute_seat_weat(target, attribute):
    target_embeddings = {
        'Arab': target[target['culture'] == 'Arab']['embedding'].values,
        'Western': target[target['culture'] == 'Western']['embedding'].values
    }
    attribute_embeddings = {
        'Positive': attribute[attribute['sentiment'] == 'positive']['embedding'].values,
        'Negative': attribute[attribute['sentiment'] == 'negative']['embedding'].values
    }

    M = target_embeddings['Arab']  
    F = target_embeddings['Western']
    A = attribute_embeddings['Positive']
    B = attribute_embeddings['Negative']

    s_m = np.array([compute_association(m, A, B) for m in M])
    s_f = np.array([compute_association(f, A, B) for f in F])

    mu_m, sigma_m = np.mean(s_m), np.std(s_m)
    mu_f, sigma_f = np.mean(s_f), np.std(s_f)

    pooled_std = np.sqrt(((len(s_m) - 1) * sigma_m**2 + (len(s_f) - 1) * sigma_f**2) / (len(s_m) + len(s_f) - 2))
    score = (mu_m - mu_f) / pooled_std

    return {
        "Score": float(score),
        "Mean Arab (M)": float(mu_m),
        "Std Arab (M)": float(sigma_m),
        "Mean Western (F)": float(mu_f),
        "Std Western (F)": float(sigma_f)
    }

def compute_same(target, attribute):
    target_embeddings = {
        'Arab': target[target['culture'] == 'Arab']['embedding'].values,
        'Western': target[target['culture'] == 'Western']['embedding'].values
    }
    attribute_embeddings = {
        'Positive': attribute[attribute['sentiment'] == 'positive']['embedding'].values,
        'Negative': attribute[attribute['sentiment'] == 'negative']['embedding'].values
    }
    same_results = {}
    for culture in ['Arab', 'Western']:
        # Normalize attribute sets A and B
        mean_A = np.mean(attribute_embeddings['Positive'], axis=0)
        mean_B = np.mean(attribute_embeddings['Negative'], axis=0)
        normalized_A = mean_A / np.linalg.norm(mean_A)
        normalized_B = mean_B / np.linalg.norm(mean_B)

        # Calculate SAME score
        same_score = 0
        for t in target_embeddings[culture]:
            b_t = np.dot(t, normalized_A - normalized_B)
            same_score += abs(b_t)

        same_results[culture] = same_score / len(target_embeddings[culture])
        
    return same_results 

def process_same_data(same_data, target):
    overall_key = f"SAME-{target}"
    entities = [k for k in same_data if k != overall_key] + ["Total SAME"]

    same_data_processed = {
        "Arab": [
            same_data[overall_key]["Arab"] if entity == "Total SAME" else same_data[entity]["Arab"]
            for entity in entities
        ],
        "Western": [
            same_data[overall_key]["Western"] if entity == "Total SAME" else same_data[entity]["Western"]
            for entity in entities
        ]
    }    
    entities = [(e.get_text() if hasattr(e, 'get_text') else e).replace(" ", "\n").replace("-", "\n") for e in entities]

    return entities, same_data_processed

def process_seat_weat_data(combined_results, culture_embeddings, sentence_embeddings):
    weat_data = {
        entity: combined_results[f"WEAT for {entity}"]["Score"]
        for entity in culture_embeddings['entity'].unique()}
    weat_data['Total WEAT'] = combined_results["WEAT"]["Score"]

    seat_data = {
        entity: combined_results[f"SEAT for {entity}"]["Score"]
        for entity in sentence_embeddings['entity'].unique()}
    seat_data['Total SEAT'] = combined_results["SEAT"]["Score"]

    global_max = max(max(list(weat_data.values())), max(list(seat_data.values())))
    global_min = min(min(list(weat_data.values())), min(list(seat_data.values())))
    
    entities = [e for e in weat_data.keys() if e != "Total WEAT"] + ["Total"]

    weat_scores = [weat_data["Total WEAT"] if e == "Total" else weat_data[e] for e in entities]
    seat_scores = [seat_data["Total SEAT"] if e == "Total" else seat_data[e] for e in entities]

    entities = [(e.get_text() if hasattr(e, 'get_text') else e).replace(" ", "\n").replace("-", "\n") for e in entities]
    scores = {"WEAT": weat_scores, "SEAT": seat_scores}

    return entities, scores, global_max, global_min

def grouped_barplot(group_names, scenario_data, title="", ylabel="", 
                    fig_size=(16, 3), hline=None, vline=False, colors=None, ylim=None,
                    output_png=None, output_eps=None):
    groups = group_names
    scenario_keys = list(scenario_data.keys())
    n_groups = len(groups)
    n_scenarios = len(scenario_keys)
    
    total_width = 0.8
    bar_width = total_width / n_scenarios
    x = np.arange(n_groups)
    
    fig, ax = plt.subplots(figsize=fig_size)

    if ylim is not None:
        max_val = max(max(vals) for vals in scenario_data.values() if vals)
        min_val = min(min(vals) for vals in scenario_data.values() if vals)

        min_headroom = abs(ylim[1] - ylim[0]) * 0.15 if (min_val == ylim[0] and min_val < 0) else 0
        max_headroom = abs(ylim[1] - ylim[0]) * 0.15 if max_val == ylim[1] else 0

        ylim[0] -= min_headroom
        ylim[1] += max_headroom

        ax.set_ylim(ylim[0], ylim[1])
    
    for i, key in enumerate(scenario_keys):
        x_positions = x - total_width/2 + i * bar_width + bar_width/2
        bars = ax.bar(x_positions, scenario_data[key], width=bar_width, 
                      label=key, color=colors.get(key) if colors else None)

        for bar in bars:
            height = bar.get_height()
            va = "top" if height < 0 else "bottom"
            ax.text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}",
                    ha="center", va=va, fontsize=10)
    
    if hline is not None:
        ax.axhline(y=hline, color='grey', linestyle='dotted', linewidth=1)
    
    if vline:
        all_count = len(group_names)
        monolingual_count = len(group_names)/2
        ax.axvline(x=(all_count/2) - 0.5, color='grey', linestyle='--', linewidth=1)
        if ylim:
            ax.text(monolingual_count // 2, ylim[1] * 0.9, "Monolingual", ha='center', va='center', fontsize=14, color='grey')
            ax.text(monolingual_count + ((all_count - monolingual_count) // 2), ylim[1] * 0.9, "Multilingual", ha='center', va='center', fontsize=14, color='grey')       
    
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.legend(fontsize=12)
    plt.tight_layout()
    
    if output_png:
        plt.savefig(output_png)
    if output_eps:
        plt.savefig(output_eps, format='eps')
