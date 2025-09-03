from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader(file_path="cricketers.txt", encoding="utf-8")

docs = loader.load()

print("docs", docs)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)

split_docs = splitter.split_documents(docs)

print("split_docs", split_docs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-minilm-l6-v2")

vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="chroma_db",
    collection_name="sample"
)

vector_store.add_documents(split_docs)

# print("get emb doc meta", vector_store.get(include=["embeddings", "documents", "metadatas"]))

# print("similarity ", vector_store.similarity_search(query="who is virat", k=2))

# print("similarity with score", vector_store.similarity_search_with_score(query="who is virat", k=2))

retriever = vector_store.as_retriever(search_kwargs={"k":2})

result = retriever.invoke("tell me something about Ben Stokes")

for i, j in enumerate(result):
    print(i, j)