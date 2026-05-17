# utility functions

# imports
import re 
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print(client)

system_prompt = '''
You are a book search query enhancer. Your task is to expand a user's raw search query into a dense, keyword-rich paragraph that mimics a real book catalog entry, optimized for semantic similarity matching.

Given the user's query, infer and include the following:
- Authors: List 2-3 plausible author names who would write this type of book
- Category: Pick the single most relevant category from this list:
  Fiction, Juvenile Fiction, Biography & Autobiography, History, Literary Criticism, Philosophy, Comics & Graphic Novels,
  Religion, Drama, Juvenile Nonfiction, Poetry, Literary Collections, Science, Business & Economics, Social Science,
  Performing Arts, Cooking, Art, Body Mind & Spirit, Psychology, Travel, Computers, Self-Help, Political Science,
  Family & Relationships, Language Arts & Disciplines, Health & Fitness, Humor, Children's stories, Education
- Description: 2-3 sentences describing likely themes, plot, tone, and setting
- Num_pages: A plausible page count for the category and type of book

Rules:
- Return ONLY a single plain text paragraph, no labels, no JSON, no bullet points
- Be as specific and keyword-dense as possible — the output is embedded for cosine similarity
- If the query is vague, cast a wide net and include broad relevant terms for that genre
- Mirror the vocabulary and structure of real book catalog entries
'''

# OpenAI API call
def call_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    
    # For debugging: output of gpt
    print(response.choices[0].message.content)
    
    return response.choices[0].message.content

# Configure prompt using user_input (call this function in the recommender pipeline!)
def enhance_text(user_input):
    
    prompt = f'''
        User search query: "{user_input}"
        Expand this query into a rich descriptive paragraph using the fields: 
        Authors, Category, Description, and Published_year.
    '''
    return call_openai(prompt)

