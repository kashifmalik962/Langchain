from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import BaseMessage
from dotenv import load_dotenv
import os
import shutil
from langchain.schema import Document
import tempfile
from utils.util import get_loader

app = FastAPI()

# Allow all origins (for testing). You can restrict later to your frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or set ["http://127.0.0.1:5500", "https://hired60.co.in"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load Environment Variables
load_dotenv()
PORT = int(os.getenv("PORT", 3038))

# Chat model
chat_model = ChatOpenAI(
    model="deepseek/deepseek-r1-distill-llama-70b:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.2
)

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Persistent Chroma DB
PERSIST_DIR = "chroma_hired60_db"
COLLECTION_NAME = "hired60"

def init_vector_store():
    """Re-initialize Chroma vector store"""
    return Chroma(
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )

# Global store (single instance reused across API calls)
vector_store = init_vector_store()

# In-memory store for per-user chat history (with max 5 messages)
store = {}
MAX_HISTORY = 5


class LimitedChatMessageHistory(ChatMessageHistory):
    """ChatMessageHistory that keeps only the last MAX_HISTORY messages."""
    
    def add_message(self, role_or_message, content: str = None):
        # If a BaseMessage is passed (internal usage)
        if isinstance(role_or_message, BaseMessage):
            super().add_message(role_or_message)
        else:
            super().add_message(role_or_message, content)
        
        # Trim history if it exceeds MAX_HISTORY
        while len(self.messages) > MAX_HISTORY:
            self.messages.pop(0)


def get_session_history(session_id: str) -> LimitedChatMessageHistory:
    """Get or create chat history for a given session_id"""
    if session_id not in store:
        store[session_id] = LimitedChatMessageHistory()
    return store[session_id]



@app.post("/re-train-chatbot")
async def re_train_chatbot(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(await file.read())
            tmp_path = tmp_file.name

        loader = get_loader(tmp_path, suffix)
        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        split_docs = splitter.split_documents(documents)

        print("len(split_docs)", len(split_docs))
        for i, doc in enumerate(split_docs[:5]):
            print(f"Doc {i}: {doc.page_content[:100]}")

        # Add safely to Chroma
        try:
            vector_store.add_documents(split_docs)
            vector_store.persist()
        except Exception as e:
            print("Fallback to add_texts:", e)
            texts = [doc.page_content for doc in split_docs]
            metadatas = [doc.metadata for doc in split_docs]
            vector_store.add_texts(texts, metadatas=metadatas)
            vector_store.persist()

        # Clean up
        file.file.close()
        os.remove(tmp_path)

        return JSONResponse(status_code=200, content={
            "message": f"{len(split_docs)} chunks added to Chroma DB"
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "message": f"Failed to train chatbot: {e}"
        })



@app.get("/query-chatbot")
async def query_chatbot(query: str, session_id: str):
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        docs_page_content = [d.page_content for d in docs]
        print("ret docs", docs_page_content)

        context = "\n".join(docs_page_content)

        # Define prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
                You are Hired60's official AI assistant for a job portal platform. 
                Your purpose is to help candidates and recruiters with information related to Hired60 only.  

                ### Rules:
                - Use ONLY the provided context and conversation history to answer.  
                - If the answer is not clearly supported by the context, respond strictly with:  
                "my knowledge from Hired60’s training data is not sufficient to answer this question." 
                - Do NOT guess, assume, or provide information outside Hired60.  
                - If the user asks about the "previous question", look up the most recent user question from history.  
                If no previous question exists, reply:  
                "There is no previous question in the conversation history."  

                ### Context:
                {context}
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])


        # Create LCEL chain
        chain = prompt | chat_model

        # Wrap with history
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )

        # Debug: Print history before invoking
        history = get_session_history(session_id).messages
        print(f"History for session {session_id}: {history}")

        # Call chain with history
        response = chain_with_history.invoke(
            {"input": query, "context": context},
            config={"configurable": {"session_id": session_id}}
        )

        return JSONResponse(status_code=200, content={
            "query": query,
            "session_id": session_id,
            "answer": response.content
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "message": f"Failed to get response from chatbot: {e}"
        })




if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)