# AI Research Assistant

A full-stack AI application that lets you upload documents and ask questions about them using Retrieval-Augmented Generation (RAG), an autonomous AI agent, and a conversational web interface.

Built as a hands-on learning project covering the modern AI stack: LLMs, RAG, vector databases, AI agents, MCP, and Streamlit.

---

## Features

- **Document Q&A** — Upload PDFs, DOCX, CSV, or TXT files and ask natural language questions
- **Web Ingestion** — Index any webpage by pasting its URL
- **Persistent Vector Storage** — Documents are embedded and stored in ChromaDB; survives app restarts
- **AI Agent with Tool Selection** — Autonomously decides which of 8 tools to use per question
- **Web Search** — Integrated Tavily search for real-time external information
- **Wikipedia Search** — Instant factual lookups from Wikipedia
- **Dictionary Lookup** — Word definitions, pronunciation, and usage examples (no API key needed)
- **Summarizer** — Condenses any long text into concise bullet points
- **Translator** — Translates text to any language
- **Keyword Extractor** — Pulls the top 10 topics from any piece of text
- **Citations** — Shows which document chunks were used to generate each answer
- **Agent Reasoning Trace** — Expandable view of every tool the agent called and why
- **MCP Server** — Exposes all tools over the Model Context Protocol for use with Claude Desktop or other MCP clients

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | [Streamlit](https://streamlit.io) |
| LLM | [Groq](https://groq.com) — Llama 3.3 70B (free tier) |
| RAG Framework | [LlamaIndex](https://www.llamaindex.ai) |
| Embeddings | `BAAI/bge-small-en-v1.5` via HuggingFace (local, free) |
| Vector Database | [ChromaDB](https://www.trychroma.com) (persistent, local) |
| Web Search | [Tavily](https://tavily.com) (free tier) |
| Wikipedia | [wikipedia](https://pypi.org/project/wikipedia/) Python package (free, no key) |
| Dictionary | [Free Dictionary API](https://dictionaryapi.dev) (free, no key) |
| MCP Server | [FastMCP](https://github.com/jlowin/fastmcp) |
| Package Manager | [uv](https://github.com/astral-sh/uv) |

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              Streamlit (Frontend)            │
│   Chat UI · File Upload · Source Viewer     │
├─────────────────────────────────────────────┤
│         ReActAgent (LlamaIndex 0.14)        │
│   Decides which tool to use per question    │
├────────────────────────────────────────────────────────────────┤
│                        8 Tools                                 │
│  document_search │ web_search  │ wikipedia_search │ calculator │
│  dictionary_lookup │ summarize │ translate │ extract_keywords  │
├────────────────────────────────────────────────────────────────┤
│          ChromaDB  (Persistent Vectors)     │
│    Chunks · Embeddings · Metadata           │
├─────────────────────────────────────────────┤
│         Groq API  (Llama 3.3 70B)          │
└─────────────────────────────────────────────┘
```

**RAG Pipeline (5 steps):**
```
Upload file → Chunk text → Embed chunks → Store in ChromaDB → Retrieve on query → Generate answer
```

---

## Project Structure

```
ai-research-assistant/
├── app.py            # Main Streamlit web app (RAG + Agent + UI)
├── agent.py          # Standalone agent demo (terminal)
├── rag.py            # Standalone RAG demo (terminal)
├── chat.py           # Standalone LLM chat demo (terminal)
├── mcp_server.py     # MCP tool server (FastMCP)
├── data/
│   └── ai_overview.txt   # Sample document to test with
├── pyproject.toml    # Project dependencies
└── .env.example      # Environment variable template
```

---

## Setup

### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-research-assistant.git
cd ai-research-assistant
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Get API keys

| Service | Free Tier | Link |
|---|---|---|
| Groq (LLM) | Yes — no credit card | [console.groq.com](https://console.groq.com) |
| Tavily (Web Search) | Yes — 1000 searches/month | [app.tavily.com](https://app.tavily.com) |

### 4. Configure environment

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
```

### 5. Run the web app

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

### Web App (`app.py`)

1. Upload files (PDF, DOCX, CSV, TXT) or paste a URL in the sidebar
2. Click **Index** — documents are chunked, embedded, and stored in ChromaDB
3. Ask questions in the chat box
4. Expand **Agent steps** to see which tools were used
5. Expand **Sources** to see which document chunks were cited

### Terminal demos

```bash
# Basic LLM chat
uv run python chat.py

# RAG over documents in ./data/
uv run python rag.py

# Agent with document search + calculator
uv run python agent.py
```

### MCP Server

The MCP server exposes all 8 tools over the Model Context Protocol:

```bash
# Run the MCP server
uv run python mcp_server.py

# Inspect tools via browser UI
uv run fastmcp dev inspector mcp_server.py

# List available tools
uv run fastmcp list tools mcp_server.py

# Call a tool directly from the terminal
uv run fastmcp call mcp_server.py wikipedia_search --input '{"query": "Alan Turing"}'
uv run fastmcp call mcp_server.py dictionary_lookup --input '{"word": "serendipity"}'
uv run fastmcp call mcp_server.py translate --input '{"text": "Hello world", "target_language": "French"}'
```

To connect to Claude Desktop, add this to your `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "research-tools": {
      "command": "uv",
      "args": ["run", "python", "/path/to/ai-research-assistant/mcp_server.py"]
    }
  }
}
```

---

## Available Tools

| Tool | Description | Needs API Key? |
|---|---|---|
| `document_search` | Searches indexed documents via RAG | No |
| `web_search` | Live web search via Tavily | Yes (Tavily) |
| `wikipedia_search` | Fetches Wikipedia summaries | No |
| `dictionary_lookup` | Word definitions and examples | No |
| `summarize` | Condenses text into bullet points | No (uses Groq) |
| `translate` | Translates text to any language | No (uses Groq) |
| `extract_keywords` | Extracts top 10 topics from text | No (uses Groq) |
| `calculator` | Evaluates math expressions | No |

---

## Concepts Covered

| Concept | Description |
|---|---|
| **LLM API** | Sending prompts and receiving completions via REST API |
| **Chat History** | Maintaining conversation context across stateless API calls |
| **Embeddings** | Converting text to vectors for semantic similarity search |
| **RAG** | Grounding LLM responses in retrieved document context |
| **Vector Database** | Storing and querying high-dimensional embedding vectors |
| **AI Agent** | Autonomous tool selection using the ReAct reasoning pattern |
| **Tool Use** | Defining Python functions as callable tools for LLMs |
| **MCP** | Exposing tools over the Model Context Protocol standard |
| **Async Python** | `asyncio` / `await` for non-blocking agent execution |
| **Streamlit** | Building interactive ML apps in pure Python |

---

## License

MIT
