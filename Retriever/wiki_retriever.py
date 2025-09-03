from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang="en")

docs = retriever.invoke("tell me something about noise pollution")

print(len(docs))
for i, j in enumerate(docs):
    print(i, j.page_content)

