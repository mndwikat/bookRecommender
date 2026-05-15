import pandas as pd
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Load pretrained model (lightweight and good default)
model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Put model in eval mode
model.eval()


#Text Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Tokenizer
def get_embedding(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling (standard way to get sentence embedding)
    embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.squeeze().numpy()


def restructure_df(df):
    
     # rows to keep for tokenization
    df = df[['title', 'authors', 'categories', 'description', 'published_year', 'average_rating','num_pages', 'ratings_count']]
    
    # fill na string values
    df[['title', 'authors', 'categories', 'description']] = (
        df[['title', 'authors', 'categories', 'description']].fillna('')
    )
    
    # fill na numerical values
    df[['published_year', 'average_rating', 'num_pages', 'ratings_count']] = (
        df[['published_year', 'average_rating', 'num_pages', 'ratings_count']].fillna(0)
    )
    
    # Cancatenate columns for tokenization and embedding
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
    
    return df

def create_embedded_DB():
    
    # loading dataframe
    df = pd.read_csv("data/books.csv", delimiter=',')
    
    # restructure df for tokenization
    df = restructure_df(df)
    

    # cleaning text
    df['text'] = df['text'].apply(clean_text)
    
    # Tokeniation and Embedding
    df['bert_embeddings'] = [
        get_embedding(text) for text in df['text'].tolist()
    ]
    
    # save embeddings as numpy file
    embeddings = np.vstack(df['bert_embeddings'].values)
    np.save("data/bert_embeddings.npy", embeddings)

# Create embedded database
# create_embedded_DB()

# load it later
# embeddings = np.load("data/bert_embeddings.npy")

