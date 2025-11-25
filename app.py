from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import gradio as gr
import pandas as pd
import streamlit as st



if __name__ == '__main__':
    # Load an embedding model (you can use any embedding model)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    movies = pd.read_csv("data/imdb_top_1000.csv")
    movies = movies[['Series_Title', 'Overview']]
    #print(movies.shape)

    # Provide your own dataset here. Decide on a chunking strategy and implement here.
    texts = list(movies['Overview'])
    #print(texts)

    # Generate embeddings
    embeddings = model.encode(texts)

    # Store embeddings in ChromaDB
    #client = chromadb.Client(Settings(anonymized_telemetry=False))
    #collection = client.create_collection("my_movie_collection")
    #collection.add(documents=texts, embeddings=embeddings.tolist(), ids=[str(i) for i in range(len(texts))])

    #save ChromaDB
    #client = chromadb.PersistentClient(path="./chromadb")
    #collection = client.get_or_create_collection("my_movie_collection")
    #collection.add(documents=texts, embeddings=embeddings.tolist(), ids=[str(i) for i in range(len(texts))])

    #load from saved ChromaDB
    client = chromadb.PersistentClient(path="./chromadb")
    collection = client.get_or_create_collection("my_movie_collection")

    # Define a search function
    def semantic_search(query):
        query_embedding = model.encode([query])
        results = collection.query(query_embeddings=query_embedding.tolist(), n_results=3)
        results['title'] = []
        for overview in results['documents'][0]:
            results['title'].append(movies[movies['Overview'] == overview]['Series_Title'].values[0])
        #print("Results", results)
        return "\n\n".join([f"{title}: {doc}" for title, doc in zip(results['title'], results['documents'][0])])

    st.title("Semantic Search Engine")
    st.write("Search over your custom dataset using semantic similarity.")

    query = st.text_input("Enter your search query")

    if query:
        results = semantic_search(query)
        st.text_area("Top Matches", results, height=300)

    print("done")

