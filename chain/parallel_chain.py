from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
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

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answer from the following text \n {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="merge the provided notes and question and answer into a single document \n notes => {notes} and {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

paralled_chain = RunnableParallel({
    "notes": prompt1 | model | parser,
    "quiz": prompt2 | model | parser
})

merge_chain = prompt3 | model | parser

chain = paralled_chain | merge_chain

text = """Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It tries to find the best boundary known as hyperplane that separates different classes in the data. It is useful when you want to do binary classification like spam vs. not spam or cat vs. dog.

The main goal of SVM is to maximize the margin between the two classes. The larger the margin the better the model performs on new and unseen data. Key Concepts of Support Vector Machine
Hyperplane: A decision boundary separating different classes in feature space and is represented by the equation wx + b = 0 in linear classification.
Support Vectors: The closest data points to the hyperplane, crucial for determining the hyperplane and margin in SVM.
Margin: The distance between the hyperplane and the support vectors. SVM aims to maximize this margin for better classification performance.
Kernel: A function that maps data to a higher-dimensional space enabling SVM to handle non-linearly separable data.
Hard Margin: A maximum-margin hyperplane that perfectly separates the data without misclassifications.
Soft Margin: Allows some misclassifications by introducing slack variables, balancing margin maximization and misclassification penalties when data is not perfectly separable."""

result = chain.invoke({"text": text})

print(result)

print(chain.get_graph().draw_ascii())