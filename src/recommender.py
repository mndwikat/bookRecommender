# Logic for comparing tokenized inputs, computing a score and returning the top N results

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils import enhance_text

# import functions
from preprocessing import get_embedding

# replace the return string with the actual prompt (fetching from frontend)
def get_user_prompt():
    return "I want a book about crocodiles"

#Compares the similarties between a vector and an embedding of vectors, 
# and outputs an N number of vectors with the highest similarity to the input vector
#Uses threshold as a similarity strictness criteria, where 1 is identical

def find_similarities (prompt_embedding, db_embeddings, threshold = 0.5):
    similarity_score = cosine_similarity(prompt_embedding.reshape(1, -1), db_embeddings).flatten() #Cosine similarity to rank vectors by similarity
    
    # get indices for rows that are similar to the prompt
    valid_indices = np.where(similarity_score >= threshold)[0]
    
    # get scores of the indices
    filtered_scores = similarity_score[valid_indices]

    # Organize ranking from higher to lower
    top_indices = valid_indices[filtered_scores.argsort()[::-1]]
    
    return(top_indices) # return indices


def return_recommendations():
    
    # loading databse embeddings
    db_embeddings = np.load("data/bert_embeddings.npy")

    # Enhance user prompt
    enhanced_prompt = enhance_text(get_user_prompt())
    
    # Get embedding for user prompt
    prompt_embedding = get_embedding(enhanced_prompt)

    # Get N indices of book recommendations
    top_indices = find_similarities(prompt_embedding, db_embeddings)

    # Load database
    df = pd.read_csv("data/books.csv", delimiter=',')
    
    # return results
    results = df.iloc[top_indices]
    
    return results

# Testing
print(return_recommendations())

