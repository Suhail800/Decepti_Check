# app.py
import streamlit as st
import pandas as pd
from utils import load_deception_model, predict_deception, get_linguistic_features, sia # Import utils functions
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
# Check if 'punkt' resource is available, and download if it's not
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
except LookupError:
    # This might happen if 'punkt' is in the index but not downloaded
    nltk.download('punkt')


# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="DeceptiCheck: Deception Detection Engine", page_icon="🕵️‍♂️")

# --- Global Variables / Model Loading (Cached for performance) ---
# Use st.cache_resource to load heavy models only once
@st.cache_resource
def load_resources():
    st.spinner("Loading Deception Detection Model and Tokenizer...")
    try:
        tokenizer, model = load_deception_model()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure you have run model_training.py.")
        st.stop()

tokenizer, model, device = load_resources()

# --- CSS for styling (Optional but makes it look professional) ---
st.markdown(
    """
    <style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 2rem;
    }
    .stAlert {
        padding: 1rem;
    }
    .stSuccess {
        background-color: #e6ffe6;
        border-left: 5px solid #4CAF50;
    }
    .stError {
        background-color: #ffe6e6;
        border-left: 5px solid #f44336;
    }
    .stWarning {
        background-color: #fff9e6;
        border-left: 5px solid #ff9800;
    }
    .main-header {
        font-size: 2.5em;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 0.5em;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sub-header {
        font-size: 1.5em;
        color: #4CAF50;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.8em 1.5em;
        border-radius: 0.5em;
        border: none;
        font-size: 1.1em;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .explanation-box {
        border: 1px solid #ddd;
        padding: 1em;
        border-radius: 8px;
        margin-bottom: 1em;
        background-color: #f9f9f9;
    }
    .word-highlight-container {
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        line-height: 1.6;
        font-size: 1.1em;
        background-color: #fff;
    }
    .highlight-red { background-color: rgba(255, 99, 71, 0.6); } /* Tomato */
    .highlight-yellow { background-color: rgba(255, 255, 0, 0.6); } /* Yellow */
    .highlight-green { background-color: rgba(144, 238, 144, 0.6); } /* LightGreen */

    </style>
    """,
    unsafe_allow_html=True
)

# --- Title and Description ---
st.markdown("<h1 class='main-header'>🕵️‍♂️ DeceptiCheck: AI Deception Detection Engine 🕵️‍♀️</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <p style='text-align: center; font-size: 1.1em; color: #555;'>
    Unmasking deceptive content in social and digital texts using advanced NLP.
    </p>
    """, unsafe_allow_html=True
)

st.sidebar.header("About DeceptiCheck")
st.sidebar.info(
    """
    DeceptiCheck is an AI-powered tool designed to analyze text and predict whether it's truthful or deceptive.
    It leverages a fine-tuned DistilBERT model for state-of-the-art text classification and provides
    explainability features to help understand its predictions.

    **Key Features:**
    - Deception prediction (Truthful/Deceptive)
    - Prediction confidence scores
    - Word importance highlighting
    - Linguistic feature analysis (pronouns, readability)
    - Sentiment analysis
    """
)
st.sidebar.header("How it Works")
st.sidebar.markdown(
    """
    1.  **Input Text:** Enter any text you want to analyze.
    2.  **Model Prediction:** A DistilBERT model processes the text.
    3.  **Explainability:** The app shows why the model made its prediction by highlighting key words and providing linguistic insights.

    **Model Used:** `DistilBERT-base-uncased` fine-tuned on the "Deceptive Opinion Spam Dataset."
    """
)

# --- Main Application Logic ---
st.subheader("Enter Text for Deception Analysis:")
user_input = st.text_area(
    "Paste your text here (e.g., a review, a social media post, etc.)",
    height=250,
    placeholder="Example: This hotel was absolutely fantastic! The staff went above and beyond to make our stay comfortable and enjoyable. I highly recommend it to everyone."
)

if st.button("Analyze Text for Deception"):
    if user_input:
        with st.spinner("Analyzing text..."):
            results = predict_deception(user_input, tokenizer, model, device)

        st.markdown("<h2 class='sub-header'>Analysis Results:</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='explanation-box'>", unsafe_allow_html=True)
            st.subheader("Deception Prediction")
            
            prediction = results['prediction']
            truthful_prob = results['probabilities']['truthful']
            deceptive_prob = results['probabilities']['deceptive']

            if prediction == 'deceptive':
                st.markdown(f"<h3 style='color:red;'>Prediction: DECEPTIVE 🚨</h3>", unsafe_allow_html=True)
                st.progress(float(deceptive_prob)) # Convert to native Python float
                st.write(f"Confidence: **{deceptive_prob:.2%}** that the text is deceptive.")
            else:
                st.markdown(f"<h3 style='color:green;'>Prediction: TRUTHFUL ✅</h3>", unsafe_allow_html=True)
                st.progress(float(truthful_prob))
                st.write(f"Confidence: **{truthful_prob:.2%}** that the text is truthful.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='explanation-box'>", unsafe_allow_html=True)
            st.subheader("Sentiment Analysis")
            sentiment = results['raw_text_sentiment']
            st.write(f"**Positive:** {sentiment['pos']:.2%}")
            st.write(f"**Neutral:** {sentiment['neu']:.2%}")
            st.write(f"**Negative:** {sentiment['neg']:.2%}")
            st.write(f"**Compound Score:** {sentiment['compound']:.2f} (Overall sentiment)")
            st.markdown("</div>", unsafe_allow_html=True)


        with col2:
            st.markdown("<div class='explanation-box'>", unsafe_allow_html=True)
            st.subheader("Word Importance (Model Focus)")
            st.info("Words highlighted in **red** are more indicative of **deception**; words in **green** suggest **truthfulness**.")
            
            # Prepare for highlighting: Normalize attention scores for visual intensity
            word_importance = results['word_importance']
            
            # Filter out special tokens and handle cases where words might not be in dict
            processed_words = user_input.split()
            
            # Get max absolute attention for scaling
            max_abs_score = max(abs(score) for score in word_importance.values()) if word_importance else 1
            
            # Generate HTML for highlighting
            highlighted_text = []
            for word in processed_words:
                cleaned_word = word.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '').replace('\'', '') # Simple cleaning
                
                score = word_importance.get(cleaned_word, 0) # Default to 0 if word not found in attention
                
                # Scale score for intensity (0 to 1 for green, 0 to -1 for red)
                if score >= 0: # More truthful
                    intensity = score / max_abs_score if max_abs_score > 0 else 0
                    color_class = "highlight-green"
                else: # More deceptive
                    intensity = abs(score) / max_abs_score if max_abs_score > 0 else 0
                    color_class = "highlight-red"
                
                # Threshold for highlighting based on attention weight
                ATTENTION_THRESHOLD_PERCENTILE = 75 # Highlight words in top X percentile of attention
                
                all_scores = list(word_importance.values())
                if len(all_scores) > 0:
                    attention_threshold = np.percentile(all_scores, ATTENTION_THRESHOLD_PERCENTILE)
                else:
                    attention_threshold = 0

                score = word_importance.get(cleaned_word, 0)
                
                # Highlight if score is above threshold
                if abs(score) > attention_threshold:
                    if prediction == 'deceptive':
                        # Highlight words that contribute to the deceptive prediction (high attention) in red
                        highlighted_text.append(f"<span class='highlight-red'>{word}</span>")
                    else:
                        # Highlight words that contribute to the truthful prediction (high attention) in green
                        highlighted_text.append(f"<span class='highlight-green'>{word}</span>")
                else:
                    highlighted_text.append(word)
            
            st.markdown(f"<div class='word-highlight-container'>{' '.join(highlighted_text)}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


        st.markdown("<h2 class='sub-header'>Detailed Linguistic Features:</h2>", unsafe_allow_html=True)
        st.markdown("<div class='explanation-box'>", unsafe_allow_html=True)
        linguistic_features = results['linguistic_features']
        
        st.subheader("Basic Text Statistics")
        st.write(f"**Word Count:** {linguistic_features['word_count']}")
        st.write(f"**Sentence Count:** {linguistic_features['sentence_count']}")
        st.write(f"**Average Words per Sentence:** {linguistic_features['avg_words_per_sentence']:.2f}")
        st.write(f"**Character Count:** {linguistic_features['char_count']}")

        st.subheader("Readability Scores")
        st.write(f"**Flesch Reading Ease:** {linguistic_features['flesch_reading_ease']:.2f} (Higher = Easier)")
        st.write(f"**Dale-Chall Readability Score:** {linguistic_features['dale_chall_readability_score']:.2f} (Lower = Easier)")
        
        st.subheader("Pronoun Usage")
        st.write(f"**First Person Singular (I, me, my, myself):** {linguistic_features['first_person_singular_pronouns']}")
        st.write(f"**First Person Plural (We, us, our, ourselves):** {linguistic_features['first_person_plural_pronouns']}")
        st.write(f"**Third Person Singular (He, she, it, him, her, its):** {linguistic_features['third_person_singular_pronouns']}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<h2 class='sub-header'>Visualizations for Deception Cues:</h2>", unsafe_allow_html=True)
        
        # Plotting Linguistic Features (e.g., Pronoun usage)
        st.markdown("<div class='explanation-box'>", unsafe_allow_html=True)
        st.subheader("Pronoun Usage Distribution")
        pronoun_data = {
            'Pronoun Type': ['First Person Singular', 'First Person Plural', 'Third Person Singular'],
            'Count': [
                linguistic_features['first_person_singular_pronouns'],
                linguistic_features['first_person_plural_pronouns'],
                linguistic_features['third_person_singular_pronouns']
            ]
        }
        pronoun_df = pd.DataFrame(pronoun_data)
        fig_pronoun, ax_pronoun = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Pronoun Type', y='Count', data=pronoun_df, palette='viridis', ax=ax_pronoun)
        ax_pronoun.set_title('Pronoun Usage in Text')
        ax_pronoun.set_ylabel('Count')
        st.pyplot(fig_pronoun)
        st.caption("Deceptive texts sometimes show less first-person singular pronoun usage.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Plotting Sentiment Distribution
        st.markdown("<div class='explanation-box'>", unsafe_allow_html=True)
        st.subheader("Sentiment Distribution")
        sentiment_data = {
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Score': [
                sentiment['pos'],
                sentiment['neu'],
                sentiment['neg']
            ]
        }
        sentiment_df = pd.DataFrame(sentiment_data)
        fig_sentiment, ax_sentiment = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Sentiment', y='Score', data=sentiment_df, palette='coolwarm', ax=ax_sentiment)
        ax_sentiment.set_title('Overall Sentiment of the Text')
        ax_sentiment.set_ylabel('Score')
        st.pyplot(fig_sentiment)
        st.caption("Extremely strong or bland sentiment can sometimes be a cue.")
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>DeceptiCheck by Suhail PV</p>", unsafe_allow_html=True)