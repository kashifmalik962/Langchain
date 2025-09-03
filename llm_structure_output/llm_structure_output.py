# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional, Literal
from dotenv import load_dotenv
import os

load_dotenv()


model = ChatOpenAI(
    model="deepseek/deepseek-r1-distill-llama-70b:free",  # or "deepseek-reasoner" for DeepSeek-R1
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",  # Important!
    temperature=0.2,
    max_tokens=512,
)

class Review(BaseModel):
    summary: Optional[str] = Field(description="write down the summary")
    sentiment: Literal["pos", "neg"] = Field(description="write down the sentiment analysis")
    pros: Optional[list[str]] = Field(description="write down all the pros inside a list")
    cons: Optional[list[str]] = Field(description="write down all the cons inside a list")


structure_model = model.with_structured_output(Review)
result = structure_model.invoke("""I regret buying this product. The quality feels cheap, and it stopped working properly within just a few days. The design looks nice in pictures, but in reality it’s poorly made. Customer support was unhelpful and refused a replacement. Honestly, I feel like I wasted my money and wouldn’t recommend it to anyone.""")

print(result)