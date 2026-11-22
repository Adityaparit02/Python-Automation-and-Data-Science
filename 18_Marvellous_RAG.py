from sentence_transformers import SentenceTransformer,util
import chromadb
from ollama import chat

#   Load Pretrained Embeding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

#   Connect to existing Chroma DB
client = chromadb.PersistentClient(path="./MarvellousDB")

collection = client.get_collection(name="marvellous_knowledge")

#  Accept Question
query = input("Enter your Question")

#   Generate Query Embeding
query_embedding = model.encode(query,convert_to_tensor=True)


#   Retrive top two results
results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=2)

# Create Contxt from retried documents
documents = results["documents"][0]

context = "\n".join(documents)
print("Retrived Context : ")
print(context)