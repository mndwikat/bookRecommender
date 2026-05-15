import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#Text Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Tokenizer
def tokenize_text(text):
    """
    Tokenize text into individual words (tokens).
    
    Args:
        text (str): Input text to tokenize
    
    Returns:
        list: List of tokens (words)
    """
    # Split by whitespace and filter out empty strings
    tokens = [token for token in text.split() if token.strip()]
    return tokens

# Concatenating

def create_tokenized_DB():
    
    # loading dataframe
    df = pd.read_csv("C:\\Users\\Marah\\Desktop\\DATATHON\\bookRecommender\\data\\books.csv", delimiter=',')
    
    # rows to keep for tokenization
    df = df[['title', 'authors', 'categories', 'description', 'published_year', 'average_rating','num_pages', 'ratings_count']]
    
    df['text'] = (
        df['title'] + ' ' +
        df['authors'] + ' ' +
        df['categories'] + ' ' +
        df['description'] +' ' +
        df['published_year'].astype(str) + ' ' +
        df['average_rating'].astype(str) + ' ' +
        df['num_pages'].astype(str) + ' ' +
        df['ratings_count'].astype(str)
    )

    df['text'] = df['text'].apply(clean_text)

    # Apply tokenizer to cleaned text
    df['tokens'] = df['text'].apply(tokenize_text)
    
    return df


