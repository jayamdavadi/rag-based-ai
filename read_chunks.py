import requests
import os 
import json
import pandas as pd

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()['embeddings']
    return embedding

# a = create_embedding("My name is Jay Amdavadi")

# print(a)


jsons = os.listdir("jsons")
# print(jsons)

my_dicts = []
chunk_id= 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    embeddings = create_embedding([c["text"] for c in content["chunks"]])

    print(f"Craeting embeddings for {json_file}")
    for i, chunk in enumerate(content["chunks"]):
        # print(chunk)
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id += 1
        # chunk["embedding"] = create_embedding(chunk["text"])
        my_dicts.append(chunk)
        # print(chunk)
     
    # break

print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
print(df)