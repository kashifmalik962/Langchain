from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter


text = """Galaxy AI - Welcome to the era of mobile AI. With Galaxy S24 Ultra and One UI in your hands, you can unleash completely new levels of creativity, productivity and possibility.
Titanium Frame - Meet Galaxy S24 Ultra, the ultimate form of Galaxy Ultra built with a new titanium exterior and a 17.25cm flat display. It is an absolute marvel of design.
Epic Camera - With the most megapixels on a Galaxy smartphone ever (200MP) and AI processing, Galaxy S24 Ultra sets the industry standard for image quality every time you hit the shutter. Also, the new ProVisual engine recognizes objects, improves color tone, reduces noise and brings out every detail.
Powerful Processor - Victory can be yours with the new Snapdragon 8 Gen 3 for Galaxy that gives you the power you need for all the gameplay you want. backed by an intelligent battery that keep you going longer. Also, you get to manifest graphic effects in real time with ray tracing for hyper-realistic shadows and reflections.
Built-in S Pen and Security - Experience seamless control and creativity, as the S Pen transforms your device into a versatile tool for productivity and artistry. While Knox protection and Samsung Wallet keep your data and payments safe."""

# splitter = CharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=10,
#     separator=""
#     )


splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=10
    )

docs = splitter.split_text(text)

print(docs)
print(len(docs))