# Clean text of prompt and enhance it

import re 
import os
from openai import OpenAI

# Declaring global variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def call_openai(prompt, system_prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def enhance_text(text):
    system_prompt = "You are a helpful assistant that enhances text."
    prompt = f"Enhance the following text: {text}"
    return call_openai(prompt, system_prompt)