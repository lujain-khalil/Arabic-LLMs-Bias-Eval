from utils import get_embedding, SUPPORTED_MODELS
import pandas as pd
import argparse
import os

# Load context sentences, cultural terms, and sentiment words
context_df = pd.read_csv("data/context_sentences.csv", encoding="utf-8-sig")
culture_terms_df = pd.read_csv("data/culture_terms.csv", encoding="utf-8-sig")
sentiment_terms_df = pd.read_csv("data/sentiment_words.csv", encoding="utf-8-sig")

# Set up command-line argument parser
parser = argparse.ArgumentParser(description="Extract embeddings using a specified multilingual masked LLM.")
parser.add_argument('model_name', type=str, choices=SUPPORTED_MODELS.keys(),
                    help="Name of the model to use (e.g., 'xlm-roberta-base', 'mbert', 'gigabert')")

# Parse command-line arguments
args = parser.parse_args()
MODEL_NAME = args.model_name

# Create model-specific directory to save embeddings
embeddings_dir = f"embeddings/{MODEL_NAME}/"
if not os.path.exists(embeddings_dir):
    os.makedirs(embeddings_dir)

# Generate Embeddings for Context Sentences
embeddings = []
for index, row in context_df.iterrows():
    sentence = row["Sentence"]
    culture = row["Culture"]
    entity = row["Entity"]

    try:
        embedding = get_embedding(sentence, MODEL_NAME)
        embeddings.append((culture, entity, sentence, embedding))
    except Exception as e:
        print(f"Error generating embedding for sentence: {sentence}\n{e}")

# Save Embeddings for Context Sentences
embedding_df = pd.DataFrame(embeddings, columns=["Culture", "Entity", "Sentence", "Embedding"])
embedding_df.to_pickle(f"{embeddings_dir}sentence_embeddings.pkl")

# Generate Embeddings for Cultural Terms
culture_embeddings = []
for index, row in culture_terms_df.iterrows():
    term = row["Term"]
    culture = row["Culture"]
    entity = row["Entity"]

    try:
        embedding = get_embedding(term, MODEL_NAME)
        culture_embeddings.append((culture, entity, term, embedding))
    except Exception as e:
        print(f"Error generating embedding for cultural term: {term}\n{e}")

# Save Embeddings for Cultural Terms
culture_embedding_df = pd.DataFrame(culture_embeddings, columns=["Culture", "Entity", "Term", "Embedding"])
culture_embedding_df.to_pickle(f"{embeddings_dir}culture_term_embeddings.pkl")

# Generate Embeddings for Sentiment Terms
sentiment_embeddings = []
for index, row in sentiment_terms_df.iterrows():
    term = row["Term"]
    sentiment = row["Sentiment"]

    try:
        embedding = get_embedding(term, MODEL_NAME)
        sentiment_embeddings.append((sentiment, term, embedding))
    except Exception as e:
        print(f"Error generating embedding for sentiment term: {term}\n{e}")

# Save Embeddings for Sentiment Terms
sentiment_embedding_df = pd.DataFrame(sentiment_embeddings, columns=["Sentiment", "Term", "Embedding"])
sentiment_embedding_df.to_pickle(f"{embeddings_dir}sentiment_term_embeddings.pkl")

print(f"Embeddings successfully generated and saved in '{embeddings_dir}' directory.")
