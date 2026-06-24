# ==============================
# Sentiment Analysis Project
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources (run once)
nltk.download("punkt")
nltk.download("vader_lexicon")
nltk.download("stopwords")

# ==============================
# Load Dataset  
# ==============================

# Update path if needed
df = pd.read_csv("amazon-fine-food-reviews/Reviews.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ==============================
# Basic Cleaning
# ==============================

df = df.dropna(subset=["Text"])
df = df.reset_index(drop=True)

# ==============================
# Tokenization Example
# ==============================

sample_text = df["Text"][0]
tokens = word_tokenize(sample_text)
print("Sample Tokens:", tokens[:20])

# ==============================
# VADER Sentiment Analysis
# ==============================

sia = SentimentIntensityAnalyzer()

def get_sentiment_scores(text):
    return sia.polarity_scores(text)

sentiment_scores = df["Text"].apply(get_sentiment_scores)

vaders = pd.DataFrame(sentiment_scores.tolist())
df = pd.concat([df, vaders], axis=1)

print(df[["neg", "neu", "pos", "compound"]].head())

# ==============================
# Sentiment Labeling
# ==============================

def sentiment_label(compound):
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["compound"].apply(sentiment_label)

print(df["Sentiment"].value_counts())

# ==============================
# Visualization
# ==============================

plt.figure(figsize=(8, 5))
df["Sentiment"].value_counts().sort_index().plot(
    kind="bar",
    title="Sentiment Distribution"
)
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ==============================
# Save Results
# ==============================

df.to_csv("sentiment_analysis_output.csv", index=False)
print("Sentiment analysis results saved successfully.")
