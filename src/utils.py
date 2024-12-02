from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

SUPPORTED_MODELS = {
    # Multilingual models
    "xlm-roberta-base": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
    "gigabert": "nlpaueb/legal-bert-base-uncased",
}

GREEN = '#90c926'  
PURPLE = '#5f26c9'
PALLETE = {'Arab': GREEN, 'Western': PURPLE}

# Embedding Function
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
