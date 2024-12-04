from utils import get_embedding, SUPPORTED_MODELS
import pandas as pd
import argparse
import os

# Load context sentences, cultural terms, and sentiment words
context_df = pd.read_csv("data/context_sentences.csv", encoding="utf-8-sig")
culture_terms_df = pd.read_csv("data/culture_terms.csv", encoding="utf-8-sig")
sentiment_terms_df = pd.read_csv("data/sentiment_terms.csv", encoding="utf-8-sig")

# Set up command-line argument parser
parser = argparse.ArgumentParser(description="Extract embeddings using a specified multilingual masked LLM.")
parser.add_argument('model_name', type=str, choices=SUPPORTED_MODELS.keys(), help="Name of the model to use (e.g., 'xlm-roberta-base', 'mbert', 'gigabert')")

# Parse command-line arguments
args = parser.parse_args()
MODEL_NAME = args.model_name

# Create model-specific directory to save embeddings
embeddings_dir = f"embeddings/{MODEL_NAME}/"
if not os.path.exists(embeddings_dir):
    os.makedirs(embeddings_dir)

# Generate Embeddings for Context Sentences
print(f"Generating {MODEL_NAME} embeddings for context sentences...")
embeddings = []
for index, row in context_df.iterrows():
    culture = row["culture"]
    entity = row["entity"]
    sentence = row["sentence"]

    try:
        print(f"Sentence {index+1}/{context_df.shape[0]}")
        embedding = get_embedding(sentence, MODEL_NAME)
        embeddings.append((culture, entity, sentence, embedding))
    except Exception as e:
        print(f"Error generating embedding for sentence: {sentence}\n{e}")

# Save Embeddings for Context Sentences
print(f"Saving...")
embedding_df = pd.DataFrame(embeddings, columns=["culture", "entity", "sentence", "embedding"])
embedding_df.to_pickle(f"{embeddings_dir}context_sentence_embeddings.pkl")

# Generate Embeddings for Cultural Terms
print(f"Generating {MODEL_NAME} embeddings for cultural terms...")
culture_embeddings = []
for index, row in culture_terms_df.iterrows():
    culture = row["culture"]
    entity = row["entity"]
    term = row["term"]

    try:
        print(f"Culture term {index+1}/{culture_terms_df.shape[0]}")
        embedding = get_embedding(term, MODEL_NAME)
        culture_embeddings.append((culture, entity, term, embedding))
    except Exception as e:
        print(f"Error generating embedding for cultural term: {term}\n{e}")

# Save Embeddings for Cultural Terms
print(f"Saving...")
culture_embedding_df = pd.DataFrame(culture_embeddings, columns=["culture", "entity", "term", "embedding"])
culture_embedding_df.to_pickle(f"{embeddings_dir}culture_term_embeddings.pkl")

# Generate Embeddings for Sentiment Terms
print(f"Generating {MODEL_NAME} embeddings for sentiment terms...")
sentiment_embeddings = []
for index, row in sentiment_terms_df.iterrows():
    sentiment = row["sentiment"]
    term = row["term"]

    try:
        print(f"Sentiment term {index+1}/{sentiment_terms_df.shape[0]}")
        embedding = get_embedding(term, MODEL_NAME)
        sentiment_embeddings.append((sentiment, term, embedding))
    except Exception as e:
        print(f"Error generating embedding for sentiment term: {term}\n{e}")

# Save Embeddings for Sentiment Terms
print(f"Saving...")
sentiment_embedding_df = pd.DataFrame(sentiment_embeddings, columns=["sentiment", "term", "embedding"])
sentiment_embedding_df.to_pickle(f"{embeddings_dir}sentiment_term_embeddings.pkl")

print(f"Embeddings successfully generated and saved in '{embeddings_dir}' directory.")
