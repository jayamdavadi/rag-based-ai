import requests
import os 
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import openai

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()['embeddings']
    return embedding
#model : ollama run llama3.2
# def inference(prompt):
#     r = requests.post("http://localhost:11434/api/generate", json={
#         "model": "llama3.2",
#         "prompt": prompt,
#         "stream": False
#     })
#     response = r.json()
#     print(response)
#     return response
def inference(prompt):
    response = openai.chat.completions.create(
        model="gpt-4",  
        messages=[
            {"role": "system", "content": "You are a helpful programming tutor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1000
    )
    answer = response.choices[0].message.content
    print(answer)
    return answer


df = joblib.load("embeddings.joblib")


incoming_query = input("Ask the Question:")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)

# Find similarity of question_embedding with other embeddings
similarities = cosine_similarity(np.vstack(df['embedding'].values),[question_embedding]).flatten()
# print(similarities)
top_results = 5
max_index = similarities.argsort()[::-1][:top_results]
# print(max_index)
new_df = df.loc[max_index]
# print(new_df[["title","number" , "text"]])

prompt = f'''I am teaching few programming languages in 10 minutes tutoriols. Here are 
video subtitle chunks containing video title ,video number,start time in seconds, 
end time in seconds, text at that time:

{new_df[["title", "number","start_time", "end_time", "text"]].to_json(orient="records")}
-----------------------------------
"{incoming_query}"
User asked this question related to video chunks, you have to answer in human way(don't mention the 
above formate, it's just for you) where and how much content is taught where 
(in which video and what timestamp) and guide the user to go to that perticular video.
If user ask unrelated questions tell him that you can only answer questions related to the course.
'''
with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)
with open("response.txt", "w") as f:
    f.write(response)

# for index, item in new_df.iterrows():
#     print(index,item["title"], item["number"], item["text"],item["start"],item["end"])
