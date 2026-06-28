import os
import asyncio
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# --- Setup LLM and Embeddings ---
Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# --- Build the RAG index ---
print("Loading and indexing documents...")
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)
print("Done!\n")

# --- Tool 1: RAG search ---
rag_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="document_search",
    description=(
        "Use this to search through the uploaded documents. "
        "Input should be a question about AI, machine learning, RAG, or related topics."
    )
)

# --- Tool 2: Calculator ---
def calculate(expression: str) -> str:
    """Evaluates a basic math expression like '15 * 47' or '100 / 4'"""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Could not calculate: {e}"

calc_tool = FunctionTool.from_defaults(
    fn=calculate,
    name="calculator",
    description="Use this to evaluate math expressions. Input must be a valid math expression."
)

# --- Create the Agent ---
# LlamaIndex 0.14+ uses async agents — asyncio.run() bridges sync and async
agent = ReActAgent(
    tools=[rag_tool, calc_tool],
    llm=Settings.llm,
    verbose=True
)

async def ask(question: str) -> str:
    response = await agent.run(question)
    return str(response)

print("Agent ready! Ask anything. Type 'quit' to exit.\n")
while True:
    question = input("You: ")
    if question.lower() == "quit":
        break

    answer = asyncio.run(ask(question))
    print(f"\nFinal Answer: {answer}\n")
    print("-" * 50 + "\n")
