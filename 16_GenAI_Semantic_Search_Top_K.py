from sentence_transformers import SentenceTransformer,util


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



