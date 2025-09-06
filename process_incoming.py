import requests
import os 
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()['embeddings']
    return embedding

# a = create_embedding("My name is Jay Amdavadi")
df = joblib.load("embeddings.joblib")
# print(a)


incoming_query = input("Ask the Question:")
question_embedding = create_embedding(incoming_query)[0]
# print(question_embedding)

# Find similarity of question_embedding with other embeddings
similarities = cosine_similarity(np.vstack(df['embedding'].values),[question_embedding]).flatten()
print(similarities)
top_results = 3
max_index = similarities.argsort()[::-1][0:top_results]
print(max_index)
new_df = df.loc[max_index]
print(new_df[["title","number" , "text"]])