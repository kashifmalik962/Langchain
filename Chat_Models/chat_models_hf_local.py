from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={"device_map": "cpu"},  # force CPU
    pipeline_kwargs={
        "temperature":0.3,
        "max_new_tokens": 50
    }
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Convert this English sentence into Hinglish (Roman Hindi): 'What is your name?'")
print(result.content)