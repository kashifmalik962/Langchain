from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Debugging check
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

llm = OpenAI(model="gpt-3.5-turbo-instruct") # api_key=OPENAI_API_KEY

result = llm.invoke("what is the capital of India")

print(result)