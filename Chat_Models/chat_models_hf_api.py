from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # or another model
    task="text-generation",
    temperature=0.2,
    max_new_tokens=5
)

model = ChatHuggingFace(llm=llm)
response = model.invoke("What is the capital of India? Answer in one word only.")
print(response.content)

