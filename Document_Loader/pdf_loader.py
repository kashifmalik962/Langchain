from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

parser = StrOutputParser()



model = ChatOpenAI(
    model="deepseek/deepseek-r1-distill-llama-70b:free",  # or "deepseek-reasoner" for DeepSeek-R1
    openai_api_key= os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",  # Important!
    temperature=0.2,
    max_tokens=512,
)

template = PromptTemplate(
    template="Answer the following question \n {question} from the following text - \n {text}",
    input_variables=["question", "text"]
)


loader = WebBaseLoader("https://www.amazon.in/Samsung-Galaxy-Smartphone-Titanium-Storage/dp/B0CS5XW6TN?ref_=ast_slp_dp")

docs = loader.load()

print("docs loaded")

chain = template | model | parser

result = chain.invoke({"question": "what is configuaration of model", "text": docs[0].page_content})

print(result)
print(type(result))



# loader = PyPDFLoader(
#     file_path="102_resume.pdf"
# )

# documents = loader.load()

# print(f"Total docs loaded: {len(documents)}")
# print(documents[0].page_content)




# from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# loader = DirectoryLoader(
#     path=r"C:\Users\kashif\Desktop\Langchain\Document_Loader",
#     glob="*.pdf",              # match only PDFs
#     loader_cls=PyPDFLoader
# )

# documents = loader.load()

# print(f"Total docs loaded: {len(documents)}")
# print(documents)

# for docs in documents:
#     print(docs.page_content)