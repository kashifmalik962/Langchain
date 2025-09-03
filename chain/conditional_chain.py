from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="deepseek/deepseek-r1-distill-llama-70b:free",  # or "deepseek-reasoner" for DeepSeek-R1
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",  # Important!
    temperature=0.2,
    max_tokens=1024,
)

class Feedback(BaseModel):
    sentiment: Literal["postive", "negative"] = Field(description="give the sentiment of the feedback")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate the sentiment postive and negative feedback from this text \n {text}",
    input_variables=["text"]
)

structure_model = model.with_structured_output(Feedback)

classifier_chain = prompt1 | structure_model

prompt2 = PromptTemplate(
    template="Write an appropriate respone to this positive feedback \n {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Write an appropriate respone to this negative feedback \n {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == "postive", prompt2 | model | parser ),
    (lambda x:x.sentiment == "negative", prompt3 | model | parser ),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"text": "this is bad phone"})

print(result)
print(chain.get_graph().draw_ascii())