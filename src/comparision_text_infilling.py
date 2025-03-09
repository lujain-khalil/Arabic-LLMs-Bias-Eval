import os
import glob
import random
import pandas as pd
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
import matplotlib.pyplot as plt
from utils import BASE_ENTITY_FILES, SUPPORTED_MODELS, grouped_barplot
import seaborn as sns

# -------------------------------
# Helper Functions
# -------------------------------
def load_entity_dict(data_dir, entity_files):
    """
    Loads entity terms from Excel files in the specified directory.
    Returns a dictionary mapping each entity type to a list of unique terms.
    """
    entity_dict = {}
    for entity_type, filename in entity_files.items():
        file_path = os.path.join(data_dir, filename)
        try:
            df = pd.read_excel(file_path)
            if "Entity" in df.columns:
                entity_dict[entity_type] = df["Entity"].dropna().unique().tolist()
            else:
                print(f"Warning: 'Entity' column not found in {filename}, skipping...")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return entity_dict

def load_masked_sentences(csv_file):
    """
    Loads a CSV file containing masked sentences (expects 'sentence' and 'entity' columns).
    Returns a list of (sentence, entity) tuples.
    """
    df = pd.read_csv(csv_file, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    masked_sentences = list(zip(df['sentence'], df['entity']))
    return masked_sentences

def filter_predictions(sentence, entity_type, entity_dict, fill_mask, num_predictions):
    mask_token = fill_mask.tokenizer.mask_token
    if mask_token not in sentence:
        sentence = sentence.replace("[MASK]", mask_token)

    preds = fill_mask(sentence)[:num_predictions]
    predicted_words = [pred['token_str'] for pred in preds]
    filtered = [word for word in predicted_words if word in entity_dict.get(entity_type, [])]
    if len(filtered) < num_predictions:
        filtered.extend(random.choices(entity_dict.get(entity_type, []), k=num_predictions - len(filtered)))
    return filtered[:num_predictions]

def get_filtered_predictions_grouped(sentences, entity_dict, fill_mask, num_predictions=50):
    grouped = {key: [] for key in entity_dict.keys()}
    for sentence, entity_type in sentences:
        grouped[entity_type].extend(filter_predictions(sentence, entity_type, entity_dict, fill_mask, num_predictions))
    return grouped

def get_filtered_predictions_by_sentence(sentences, entity_dict, fill_mask, num_predictions=50):
    return [filter_predictions(sentence, entity_type, entity_dict, fill_mask, num_predictions)
            for sentence, entity_type in sentences]

# -------------------------------
# Comparison Graph
# -------------------------------

def build_culture_dict(excel_folder):
    excel_files = glob.glob(os.path.join(excel_folder, "*.xlsx"))
    culture_dict = {}
    for file in excel_files:
        df = pd.read_excel(file)
        for _, row in df.iterrows():
            entity = str(row["Entity"]).strip()
            culture = str(row["Culture"]).strip()
            culture_dict[entity] = culture
    return culture_dict

def get_scenario_data(entity_culture_dict, text_infilling_path):
    scenarios = ["agnostic", "contextualized"]
    scenario_arab = {s.capitalize(): [] for s in scenarios}

    for s in scenarios:
        scenario_dir = os.path.join(text_infilling_path, s)
        for model in SUPPORTED_MODELS.keys():
            csv_file = os.path.join(scenario_dir, f"{model}_masked_predictions.csv")
            df = pd.read_csv(csv_file, encoding="utf-8-sig")
            # Combine all predictions from all columns.
            all_preds = []
            for col in df.columns:
                all_preds.extend(df[col].dropna().astype(str).tolist())
            arab_count = sum(1 for pred in all_preds if entity_culture_dict.get(pred, "Unknown") == "Arab")
            western_count = sum(1 for pred in all_preds if entity_culture_dict.get(pred, "Unknown") == "Western")
            total = arab_count + western_count
            if total == 0:
                scenario_arab[s.capitalize()].append(0)
            else:
                scenario_arab[s.capitalize()].append((arab_count / total))
    return scenario_arab

# -------------------------------
# Main Pipeline Function
# -------------------------------
def main(data_dir, sentences_csv, model_name, scenario="agnostic", output_mode="grouped",
         num_predictions=50, output_csv="predictions.csv"):
    
    # Omit "religious places" in contextual mode.
    entity_files = (BASE_ENTITY_FILES if scenario.lower() == "agnostic"
                    else {k: v for k, v in BASE_ENTITY_FILES.items() if k != "religious places"})
    
    entity_dict = load_entity_dict(data_dir, entity_files)
    masked_sentences = load_masked_sentences(sentences_csv)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=num_predictions)

    if output_mode.lower() == "grouped":
        predictions = get_filtered_predictions_grouped(masked_sentences, entity_dict, fill_mask, num_predictions)
        df_predictions = pd.DataFrame({k: pd.Series(v) for k, v in predictions.items()})
    elif output_mode.lower() == "by_sentence":
        predictions = get_filtered_predictions_by_sentence(masked_sentences, entity_dict, fill_mask, num_predictions)
        df_predictions = pd.DataFrame(predictions).transpose()
        df_predictions.columns = [s[0] for s in masked_sentences]
    else:
        print("Invalid output_mode specified. Choose 'grouped' or 'by_sentence'.")
        return

    df_predictions.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nPredictions saved to: {output_csv}")

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    # Determine base directories relative to this script's location.
    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(base_dir, "..", "data")
    entities_dir = os.path.join(data_dir, "entities")
    text_infilling_dir = os.path.join(data_dir, "text-infilling-results")

    fig_dir = os.path.join(base_dir, "..", "results", "model_comparisions")
    eps_dir = os.path.join(fig_dir, "eps")

    for dir in [data_dir, entities_dir, text_infilling_dir, fig_dir, eps_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    output_mode = "grouped"
    num_predictions = 50

    # Define scenarios as a dictionary
    scenarios = {
        "agnostic": os.path.join(data_dir, "ag_neutral_sentences.csv"),
        "contextualized": os.path.join(data_dir, "co_neutral_sentences.csv")
    }
    
    # Loop over each scenario & model combination
    # Commeent out this loop to avoid regenerating predictions every time
    for scenario_name, sentences_csv in scenarios.items():
        for model_short, model_full in SUPPORTED_MODELS.items():
            # Build output CSV path dynamically
            output_dir = os.path.join(text_infilling_dir, scenario_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_csv = os.path.join(output_dir, f"{model_short}_masked_predictions.csv")
            print("\n=================================================")
            print(f"Running: scenario='{scenario_name}', model='{model_short}'...\n")
            main(
                data_dir=entities_dir,
                sentences_csv=sentences_csv,
                model_name=model_full,
                scenario=scenario_name,
                output_mode=output_mode,
                num_predictions=num_predictions,
                output_csv=output_csv
            )
    
    culture_dict = build_culture_dict(entities_dir)
    scenario_data = get_scenario_data(culture_dict, text_infilling_dir)
    
    model_names = [("XLM-Base" if m == "XLM-RoBERTa-Base" 
                    else ("XLM-Large" if m == "XLM-RoBERTa-Large" else m))
                   for m in list(SUPPORTED_MODELS.keys())]
    
    print("Generating plot...")
    grouped_barplot(model_names, scenario_data, 
                    title="Ratio of Generated Arab Entities in Text-Infilling Experiment", 
                    ylabel="Ratio", 
                    colors=None, 
                    hline=0.50,
                    vline=True,
                    ylim=[0,1],
                    output_png=os.path.join(fig_dir, 'comparison_text_infillings.png'), 
                    output_eps=os.path.join(eps_dir, 'comparison_text_infillings.eps'))