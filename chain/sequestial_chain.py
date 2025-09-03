# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following into 5 concise bullet points:\n\n{text}",
    input_variables=["text"]
)


model = ChatOpenAI(
    model="deepseek/deepseek-r1-distill-llama-70b:free",  # or "deepseek-reasoner" for DeepSeek-R1
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",  # Important!
    temperature=0.2,
    max_tokens=1024,
)

parser = StrOutputParser()

# Convert string -> dict for second prompt
to_text_input = RunnableLambda(lambda x: {"text": x})

chain = prompt1 | model | parser | to_text_input | prompt2 | model | parser

result = chain.invoke({"topic": "unemployment of India"})
print(result)

print(chain.get_graph().draw_ascii())