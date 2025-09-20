from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# Use any embedding model (here: MiniLM)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize semantic splitter
semantic_splitter = SemanticChunker(embedding_model)

# Example long text
text = """
Artificial Intelligence is changing the world. 
It is used in healthcare, finance, and education. 
Self-driving cars are also powered by AI. 
Meanwhile, climate change remains one of the biggest global challenges. 
Governments and organizations are trying to address it through policy and innovation.
"""

# Split into meaning-based chunks
docs = semantic_splitter.create_documents([text])

for i, d in enumerate(docs, 1):
    print(f"\n--- Chunk {i} ---\n{d.page_content}")
