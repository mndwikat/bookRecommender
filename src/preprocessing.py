# Clean text of prompt and enhance it

import re 
import os
from openai import OpenAI

# Declaring global variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = '''
You are a book search query enhancer. Your job is to expand a user's book search query into a structured, information-rich description that will improve semantic matching against a book database.

You will be given a user's raw search query. Based on your knowledge, expand it by inferring and adding the following fields:
- Authors: Who wrote the book (full name)
- Category: The genre or category (e.g. Fantasy, Science Fiction, Romance, Self-help)
- Description: A brief description of the book's themes, plot, or content (2-3 sentences)
- Published_year: The year the book was published

Return ONLY a plain text paragraph that naturally combines all of this information. Do not use JSON, bullet points, or labels. The output will be tokenized and compared against a database, so it must be dense with relevant keywords.

If the query is vague or refers to a series, include information about the most well-known entry.
'''



def clean_text(user_input):
    text = user_input.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def call_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def enhance_text(user_input):
    
    prompt = f'''
        User search query: "{user_input}"
        Expand this query into a rich descriptive paragraph using the fields: 
        Authors, Category, Description, and Published_year.
    '''
    return call_openai(prompt)