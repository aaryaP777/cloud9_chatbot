import os
import cohere
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
import faiss

load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(cohere_api_key)

class CohereEmbed(Embeddings):
    def embed_documents(self, texts):
        response = co.embed(texts=texts, model="embed-english-v3.0", input_type="search_document")
        return response.embeddings

    def embed_query(self, text):
        response = co.embed(texts=[text], model="embed-english-v3.0", input_type="search_query")
        return response.embeddings[0]

def load_data(file_path):
    with open(file_path, 'r') as f:
        chunks = [line.strip() for line in f.readlines() if line.strip()]
    return [Document(page_content=chunk) for chunk in chunks]

def build_vector_index():
    data_path = "data/sample_logs.txt"
    docs = load_data(data_path)
    embedder = CohereEmbed()
    db = FAISS.from_documents(docs, embedder)
    db.save_local("vector_store")
    print("Vector store created at 'vector_store/'.")

if __name__ == "__main__":
    build_vector_index()
