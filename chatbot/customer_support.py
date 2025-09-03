"""
Customer support chatbot, this chatbot have capabilities to return response behalf of user previos 
conversation and storing user chathistory.
"""
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # or another model
    task="text-generation",
    temperature=0.2,
    max_new_tokens=5
)

model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate([
    ("system", "You are a helpful customer support chatbot"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", '{query}')
])


chat_history = []
with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())

prompt = prompt.invoke({"chat_history": chat_history,  "query": "where is my refund"})

result = model.invoke(prompt)
print(result.content)
