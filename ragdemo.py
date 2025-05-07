import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from annoy import AnnoyIndex
import cohere
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

df =  pd.read_excel("ucdp-peace-agreements-221.xlsx")




def CreateIndex(loaded_embeddings , method):
    embeds = np.array(loaded_embeddings)
    search_index = AnnoyIndex(embeds.shape[1], 'angular')  # Specify the distance metric
    for i in range(len(embeds)):
        search_index.add_item(i, embeds[i])
    search_index.build(10)  # Number of trees
    search_index.save(f'Index{method}.ann')  # Ensure 'data' directory exists
    return search_index



def search_article(query, search_index, df , Method):

    if Method == 'Mini':

          tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
          model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

          inputs = tokenizer(query, return_tensors='pt', max_length=512, truncation=True, padding=True)
          with torch.no_grad():
              outputs = model(**inputs)
              hidden_states = outputs.last_hidden_state
          query_embed = hidden_states.mean(dim=1).squeeze().numpy()

          similar_item_ids, distances = search_index.get_nns_by_vector(query_embed, 20, include_distances=True)

          # Access similar articles
          search_results = [df.iloc[i]['pa_comment'] for i in similar_item_ids]
          #results_with_scores = list(zip(search_results, distances))
          return search_results , distances



def ask_article_cohere(question, num_generations=1):

    # this is a function you call for generating the answer to the query.
    # You need a Cohere API code. which is free but with limmited calls

    co = cohere.Client("FZnpfTPjBLcPHietqOGFmLoQ4CiL8QCtSXbs6asu")

    context , _ = search_article(question, Index, df , Method)

    prompt = f"""
    Excerpt from the following article:
    {context}
    Question: {question}

    Extract the answer of the question from the text provided.
    If the text doesn't contain the answer,
    reply that the answer is not available."""

    prediction = co.generate(
        prompt=prompt,
        max_tokens=200,
        model="command-nightly",
        temperature=0.5,
        num_generations=num_generations
    )
    # return prediction.generations
    return [generation.text for generation in prediction.generations]


Method = "Mini"
query =  "tell me about peace agreements in Russia "


response = np.load('embeddingsMini.npy')
embeds = np.array(response)

Index = CreateIndex(embeds , Method) #create index
print(ask_article_cohere(query, num_generations=1))

