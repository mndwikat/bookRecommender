import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("C:\\Users\\Marah\\Desktop\\DATATHON\\bookRecommender\\data\\books.csv", delimiter=',')

#print(df.shape)
#print(df.columns)
#df.head()

df = df[['title', 'authors', 'categories', 'description', 'published_year', 'average_rating','num_pages', 'ratings_count']]
# Fill NaN values with empty strings to avoid concatenation errors
df = df.fillna('')
print(f"Shape after filling NaN: {df.shape}")
#df.head()

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

#---------------------------------Concatinating---------------------------

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

df["text"]


