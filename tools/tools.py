                                ### Built-In Tool ###

# from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

# search_tool = DuckDuckGoSearchRun()

# cmd_tool = ShellTool()

# result = search_tool.invoke("gpt-5 latest news")

# result2 = cmd_tool.invoke("powershell -Command pwd")

# print(result)
# print(result2)

#                                         ### Custom Tool ###

# from langchain_core.tools import tool

# @tool
# def multiply_two_number(a: int, b: int) -> int:   # Type 
#     """Multiply two numbers"""
#     return a*b


                                ###  Create tool using BaseTool/Parent class  ### 

# from langchain.tools import BaseTool
# from pydantic import Field, BaseModel
# from typing import Type

# class MultiplyInput(BaseModel):
#     a: int = Field(required=True, description="this first number to add")
#     b: int = Field(required=True, description="this second number to add")


# class MultiplyTool(BaseTool):
#     name: str = "multiply"
#     description: str = "give 2 numbers return product of two numbers"
#     args_schema: Type[BaseModel] = MultiplyInput

#     def _run(self, a:int, b:int) -> int:
#         return a*b


# multiply_tool = MultiplyTool()

# print(multiply_tool.name)
# print(multiply_tool.description)
# print(multiply_tool.args)

# result = multiply_tool.invoke({"a":10, "b":20})
# print(result)



                                    ### Tool binding and calling ###


from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Type
import os
from dotenv import load_dotenv

load_dotenv()

# Initialized model
llm = ChatOpenAI(
    model="openrouter/sonoma-dusk-alpha",  # or "deepseek-reasoner" for DeepSeek-R1
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",  # Important!
    temperature=0.2,
    max_tokens=1024,
)


# Create tool using BaseTool class
class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="this is first number to add")
    b: int = Field(required=True, description="this is second number to add")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "based on given number return product"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a:int, b:int) -> int:
        return a*b


multipy_tool = MultiplyTool()

# human querry
querry = HumanMessage("can you multiply 10 with 4")
messages = [querry]

# Tool binding with llm that help to understand llm which tool exist
llm_with_tools = llm.bind_tools([multipy_tool])

print("messages", messages)
result = llm_with_tools.invoke(messages)
print("result", result)
messages.append(result)
result_from_toolkit = multipy_tool.invoke(result.tool_calls[0])
print("result_from_toolkit", result_from_toolkit)
messages.append(result_from_toolkit)

final_result = llm_with_tools.invoke(messages)
print("final_result", final_result)