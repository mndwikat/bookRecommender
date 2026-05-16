# Logic for comparing tokenized inputs, computing a score and returning the top N results

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Example for testing
user_prompt_vector = np.array(
    [0.0412,  -0.1183,   0.3291,   0.0057,  -0.2748, 0.7564, 0.6584, 0.3265])

#Example for testing (8 dimensions)
database_vectors = np.array([
    [ 0.0391, -0.1201,  0.3310,  0.0071, -0.2700,  0.1812, 0.00159, 0.4378],  
    [ 0.2847,  0.0634, -0.1928,  0.3041,  0.1193, -0.0572, 0.03678, 0.2345],  
    [-0.1023,  0.4512,  0.0837, -0.2019,  0.3748,  0.2031, 0.01234, 0.3567],  
    [ 0.3312, -0.0891,  0.2743,  0.1837, -0.1204,  0.0639, 0.4764, 0.4564],
    [0.0412,  -0.1183,   0.3291,   0.0057,  -0.2748, 0.7564, 0.6584, 0.3265]  
])

#Compares the similarties between a vector and an embedding of vectors, 
# and outputs an N number of vectors with the highest similarity to the input vector

def find_similarities (vector, embeding, N_outputs):
    
    similarity_score = cosine_similarity(vector.reshape(1, -1), embeding).flatten() #Cosine similarity to rank vectors by similarity
    
    top_indices = similarity_score.argsort()[::-1][:N_outputs] #Organize ranking from higher to lowesr

    return(embeding[top_indices]) #Return corresponding vectors to rankings
        

    

print(find_similarities(user_prompt_vector, database_vectors, 3))

