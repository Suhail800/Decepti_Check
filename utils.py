# utils.py
import pandas as pd
import os
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import textstat
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Download necessary NLTK data ---
required_nltk_data = ['stopwords', 'wordnet', 'vader_lexicon', 'punkt']
for data_name in required_nltk_data:
    try:
        # Check if the data is found by NLTK's resource finder
        # These are the standard paths NLTK expects for these specific resources
        if data_name == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif data_name == 'stopwords':
            nltk.data.find('corpora/stopwords')
        elif data_name == 'wordnet':
            nltk.data.find('corpora/wordnet')
        elif data_name == 'vader_lexicon':
            nltk.data.find('sentiment/vader_lexicon')
        
        print(f"NLTK resource '{data_name}' already downloaded and found.")
    except LookupError:
        print(f"Downloading NLTK resource: {data_name}...")
        nltk.download(data_name)
        print(f"Finished downloading {data_name}.")

# --- Data Loading and Preprocessing ---
def load_data(data_path="data/deceptive-opinion-spam-corpus", file_name='deceptive-opinion.csv'):
    """
    Loads the deceptive and truthful review dataset from a single CSV file.
    Assumes the file contains 'deceptive' and 'text' columns.
    """
    full_path = os.path.join(data_path, file_name)
    if not os.path.exists(full_path):
        print(f"ERROR: Dataset file not found at: {full_path}. "
              "Please ensure '{file_name}' is in the specified data path.")
        raise FileNotFoundError(f"Dataset file not found: {full_path}")
    
    df = pd.read_csv(full_path)

    df = df.rename(columns={'deceptive': 'label'})
    df['text'] = df['text'].astype(str)
    df.dropna(subset=['text', 'label'], inplace=True)

    if 'polarity' in df.columns:
        initial_len = len(df)
        df = df[df['polarity'] == 'positive'].reset_index(drop=True)
        if len(df) < initial_len:
            print(f"Note: Dataset filtered to include only 'positive' reviews. Remaining: {len(df)} reviews.")
    
    print(f"Loaded {len(df)} reviews from '{file_name}'.")
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Linguistic Feature Extraction ---
# Ensure these are initialized AFTER the NLTK downloads are confirmed
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer() 

def get_linguistic_features(text):
    words = text.split()
    sentences = nltk.sent_tokenize(text) # This is where 'punkt' is used
    
    word_count = len(words)
    sentence_count = len(sentences)
    char_count = len(text)
    
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    dale_chall_readability_score = textstat.dale_chall_readability_score(text)
    
    first_person_singular = len(re.findall(r'\b(i|me|my|mine|myself)\b', text, re.IGNORECASE))
    first_person_plural = len(re.findall(r'\b(we|us|our|ours|ourselves)\b', text, re.IGNORECASE))
    third_person_singular = len(re.findall(r'\b(he|she|it|him|her|his|hers|its|himself|herself|itself)\b', text, re.IGNORECASE))
    
    sentiment_scores = sia.polarity_scores(text)
    
    features = {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_words_per_sentence': word_count / sentence_count if sentence_count > 0 else 0,
        'char_count': char_count,
        'flesch_reading_ease': flesch_reading_ease,
        'dale_chall_readability_score': dale_chall_readability_score,
        'first_person_singular_pronouns': first_person_singular,
        'first_person_plural_pronouns': first_person_plural,
        'third_person_singular_pronouns': third_person_singular,
        'sentiment_neg': sentiment_scores['neg'],
        'sentiment_neu': sentiment_scores['neu'],
        'sentiment_pos': sentiment_scores['pos'],
        'sentiment_compound': sentiment_scores['compound']
    }
    return features

# --- Model Loading ---
def load_deception_model(model_path="models/distilbert_deception_model", num_labels=2):
    """Loads the fine-tuned DistilBERT model and tokenizer."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    model.eval() # Set model to evaluation mode
    return tokenizer, model

# --- Prediction and Explainability ---
def predict_deception(text, tokenizer, model, device='cpu'):
    """
    Predicts deception and extracts attention weights for explainability.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    
    # 0 for truthful, 1 for deceptive (assuming this mapping from model_training.py)
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    
    attentions = outputs.attentions
    
    # Average attention from CLS token to all other tokens, averaged across heads in the last layer
    last_layer_attention = attentions[-1][0] # (num_heads, sequence_length, sequence_length)
    token_attention_weights = last_layer_attention[:, 0, :].mean(dim=0).cpu().numpy()

    # Get tokens from input_ids
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Filter out special tokens and their weights
    clean_tokens = []
    clean_weights = []
    for i, token in enumerate(tokens):
        if token not in tokenizer.all_special_tokens:
            clean_tokens.append(token)
            clean_weights.append(token_attention_weights[i])

    # Attempt to map attention weights back to original words.
    word_importance = {}
    # Use original words from the input text to ensure correct mapping
    original_words = re.findall(r'\b\w+\b', text.lower()) 
    
    # Create a mapping from subword token to its weight
    token_to_weight = {token: weight for token, weight in zip(clean_tokens, clean_weights)}
    
    # Assign weight to each original word based on its first subword token's weight (a common heuristic)
    for original_word in original_words:
        sub_tokens = tokenizer.tokenize(original_word)
        if sub_tokens:
            weights_for_word = [token_to_weight.get(t, 0) for t in sub_tokens]
            word_importance[original_word] = np.mean(weights_for_word) if weights_for_word else 0

    return {
        'prediction': 'deceptive' if predicted_class_id == 1 else 'truthful',
        'probabilities': {'truthful': probabilities[0], 'deceptive': probabilities[1]},
        'word_importance': word_importance,
        'linguistic_features': get_linguistic_features(text),
        'raw_text_sentiment': sia.polarity_scores(text)
    }

# --- Visualization for explainability (optional, could be in Streamlit too) ---
def plot_attention_heatmap(tokens, attention_weights, title="Attention Heatmap"):
    """
    Plots a heatmap of attention weights.
    (This function is not directly used in the Streamlit app but is part of utils for completeness)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, cmap="viridis", cbar=True)
    plt.xticks(np.arange(len(tokens)), tokens, rotation=90)
    plt.yticks(np.arange(len(tokens)), tokens)
    plt.title(title)
    plt.tight_layout()
    plt.show()