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

jsons = os.listdir("jsons")
# print(jsons)

my_dicts = []
chunk_id= 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    embeddings = create_embedding([c["text"] for c in content["chunks"]])

    print(f"Creating embeddings for {json_file}")
    for i, chunk in enumerate(content["chunks"]):
        # print(chunk)
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id += 1
        # chunk["embedding"] = create_embedding(chunk["text"])
        my_dicts.append(chunk)
        # if(i==5):
        #     break
    # break
        # print(chunk)
     
# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)

# save the data frame
joblib.dump(df, "embeddings.joblib")

