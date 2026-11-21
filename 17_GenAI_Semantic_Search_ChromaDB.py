from sentence_transformers import SentenceTransformer,util
import chromadb

#   Load Pretrained Embeding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

#   Knowledge Base
documents = [
    "Python is used for machine learning",
    "Java is used for enterprise application",
    "SQL is used to manage Databases",
    "HTML is used to create web pages",
    "Tensorflow is used for Deep learning"
]

#   Generate Document Embeding
document_embedding = model.encode(documents,convert_to_tensor=True)

#   Create ChromaDB client
client = chromadb.PersistentClient(path="./MarvellousDB")

#   Create Collection 
collection = client.get_or_create_collection(
    name="marvellous_knowledge",
    configuration={"hnsw":{"space":"cosine"}}
)                                                    # hnsw = Hierarchial Navigable Small World


# Store Documents and embeddings
collection.upsert(
    ids=["doc1","doc2","doc3","doc4","doc5"],
    documents=documents,
    embeddings=document_embedding.tolist()
)

print("Total Documents : ",collection.count())


#  Accept Question
query = input("Enter your Question")

#   Generate Query Embeding
query_embedding = model.encode(query,convert_to_tensor=True)

#   Calculate Similarity
scores = util.cos_sim(query_embedding,document_embedding)[0]

#   Find top two matching documents
top_result = scores.topk(2)

print("Top 2 relevant documents : ")
for index in top_result.indices : 
    print(documents[index.item()])



