from langchain_openai import ChatOpenAI
from langchain_core.tools import InjectedToolArg
from langchain_core.messages import HumanMessage
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Annotated
from dotenv import load_dotenv
import os
import requests

load_dotenv()

# Initialized model
model = ChatOpenAI(
    model="openrouter/sonoma-dusk-alpha",  # or "deepseek-reasoner" for DeepSeek-R1
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",  # Important!
    temperature=0.2,
    max_tokens=1024,
)

# Initialize tool for data fething
class Fetch_Data(BaseModel):
    base_currency: str = Field(description="This is base currency")
    target_currency: str = Field(description="This is Target currency")


class Fetch_Data_tool(BaseTool):
    name: str = "fetch_data_tool"
    description: str = "This function is responsible for fetching data of currecy using external apis"
    args_schema: Type[BaseModel] = Fetch_Data

    def _run(self, base_currency:str, target_currency:str) -> str:
        url = f"https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}"
        response = requests.get(url).json()

        if "error" in response:
            return {"error": response["error"]["message"]}
        # conversion_rate = response.get("conversion_rate")

        return response
        

# Initialize tool for currency calculation
class Calculate_Currency(BaseModel):
    base_currency_value: float = Field(description="This is base currency value")
    conversion_rate: float = Field(description="This is conversion rate")


class Converter(BaseTool):
    name: str = "converter_tool"
    description: str = "This function is responsible for convert base currency value and conversion rate"
    args_schema: Type[BaseModel] = Calculate_Currency

    def _run(self, base_currency_value:float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
        return base_currency_value*conversion_rate
    
print("args", Converter.args)

fetch_data_tool = Fetch_Data_tool()

convert = Converter()

# Bind Tools with LLM
model_with_tools = model.bind_tools([fetch_data_tool, convert])

messages = [HumanMessage("what is the conversion factor between USD and INR, and based on that can you convert 10 usd to inr")]

ai_respone = model_with_tools.invoke(messages)
print("ai_respone", ai_respone)

print("ai_respone.tool_calls", ai_respone.tool_calls)