from sentence_transformers import SentenceTransformer

#   Load Pretrained Embeding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Python is a programming language",
    "Python is used for data science",
    "I support indian cricket team"
]

embeddings = model.encode(sentences)

# Display sentence embeddings
for index , embedding in enumerate (embeddings):
    print("-"*70)
    print("\nSentence : ",sentences[index])

    print("Embeding : ")
    print(embedding)

    print("Total Dimension : ", len(embeddings))