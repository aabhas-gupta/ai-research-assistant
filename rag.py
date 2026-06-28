import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# Tell LlamaIndex which LLM and embedding model to use globally
Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"  # small, fast, free, runs locally
)

print("Loading documents...")
documents = SimpleDirectoryReader("data").load_data()
print(f"Loaded {len(documents)} document(s)\n")

# This one line does steps 2-4: chunk → embed → store
print("Building index (chunking + embedding your documents)...")
index = VectorStoreIndex.from_documents(documents)
print("Index ready!\n")

# The query engine handles steps 5-6: retrieve → generate
query_engine = index.as_query_engine(similarity_top_k=3)

print("Ask questions about your documents. Type 'quit' to exit.\n")
while True:
    question = input("You: ")
    if question.lower() == "quit":
        break

    response = query_engine.query(question)
    print(f"\nAI: {response}\n")
