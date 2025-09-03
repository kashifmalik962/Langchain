from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# text = "Delhi is the capital of India"
# result = embedding.embed_query(text)

document = ["Sachin Tendulkar: Widely recognized as one of the greatest batsmen of all time",
            "Virat Kohli: Kohli is considered one of the best batsmen of the modern era",
            "MS Dhoni: Dhoni is celebrated for his leadership and wicketkeeping skills",
            "Jasprit Bumrah: Bumrah is one of the premier fast bowlers in international cricket",
            "Mohammed Siraj: Siraj is a skilled and aggressive fast bowler who has rapidly risen in international cricket"]

embedded_document = embedding.embed_documents(document)

embedded_querry = embedding.embed_query("tell me about sachin")

scores = cosine_similarity([embedded_querry], embedded_document)[0]

print(sorted(list(enumerate(scores)), key=lambda x:x[1]))
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]


print(document[index])
print("Similarity score is :", score)