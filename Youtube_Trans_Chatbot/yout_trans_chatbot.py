from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(model="deepseek/deepseek-r1-distill-llama-70b:free",  # or "deepseek-reasoner" for DeepSeek-R1
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
    )

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

parser = StrOutputParser()

loader = TextLoader("transcript.txt")
transcript = loader.load()

print("len(transcript)", len(transcript))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
    )

split_transcript = splitter.split_documents(transcript)

print("split_transcript[0]", split_transcript[0])
print("len(split_transcript)", len(split_transcript))

unique_chunks = []
seen = set()

for doc in split_transcript:
    if doc.page_content not in seen:
        unique_chunks.append(doc)
        seen.add(doc.page_content)

vector_store = Chroma.from_documents(
    documents=unique_chunks,
    embedding=embedding_model,
    collection_name="youtube_trans",
    persist_directory="chroma_db"
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})

prompt = PromptTemplate(
    template="""you are a helpful assistant.
    Answer only from the provided transcript context.
    if the context is insuffiecient, just say you don't know.
    
    {context}
    Question: {question}""",
    input_variables=["context", "question"]
)

question = "if the topic of Sam Altman is disscussed in this video? if yes then what was disscussed."
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_promt = prompt.invoke({"context": context_text, "question": question})


result = model.invoke(final_promt)
print("result", result.content)