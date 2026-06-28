import os
import asyncio
import json
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import chromadb
from llama_index.core import VectorStoreIndex, Settings, Document, StorageContext
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

CHROMA_PATH = "./chroma_db"
SOURCES_FILE = "./indexed_sources.json"  # tracks file names across restarts

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Research Assistant", page_icon="🔍", layout="wide")

# ── Load models once, cache them so they survive Streamlit reruns ─────────────
@st.cache_resource
def load_models():
    Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)

load_models()
chroma_client = get_chroma_client()

# ── ChromaDB helpers ──────────────────────────────────────────────────────────
def get_collection():
    return chroma_client.get_or_create_collection("research_docs")

def get_index() -> VectorStoreIndex:
    collection = get_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    if collection.count() > 0:
        return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    return VectorStoreIndex([], storage_context=storage_context)

def add_to_index(documents: list[Document]):
    collection = get_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    if collection.count() > 0:
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    else:
        index = VectorStoreIndex([], storage_context=storage_context)
    for doc in documents:
        index.insert(doc)

def get_chunk_count() -> int:
    return get_collection().count()

def reset_db():
    chroma_client.delete_collection("research_docs")
    Path(SOURCES_FILE).unlink(missing_ok=True)

# ── Indexed sources list (survives restarts via JSON file) ────────────────────
def load_sources() -> list[str]:
    if Path(SOURCES_FILE).exists():
        return json.loads(Path(SOURCES_FILE).read_text())
    return []

def save_sources(sources: list[str]):
    Path(SOURCES_FILE).write_text(json.dumps(sources))

# ── File parsers ──────────────────────────────────────────────────────────────
def parse_uploaded_file(f) -> Document:
    suffix = Path(f.name).suffix.lower()
    if suffix == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(f)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix == ".docx":
        import docx2txt
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        text = docx2txt.process(tmp_path)
        os.unlink(tmp_path)
    elif suffix == ".csv":
        import pandas as pd
        import io
        df = pd.read_csv(io.BytesIO(f.read()))
        text = df.to_string(index=False)
    else:
        text = f.read().decode("utf-8", errors="ignore")
    return Document(text=text, metadata={"file_name": f.name})

def fetch_url(url: str) -> Document:
    import requests
    from bs4 import BeautifulSoup
    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return Document(text=text, metadata={"file_name": url})

# ── Tools (built fresh per query, capture log/sources via closure) ─────────────
def make_tools(query_engine, log: list, sources: list) -> list:
    """
    Returns tool functions wired to write their activity into `log` and `sources`.
    Using closure so we can read log/sources after the agent finishes.
    """

    def search_documents(query: str) -> str:
        """Search through indexed documents to find relevant information."""
        log.append(f"🔍 Searched documents for: *{query}*")
        response = query_engine.query(query)
        for node in response.source_nodes:
            score = node.score if node.score is not None else 1.0
            # Score is cosine distance: lower = more similar.
            # Only surface chunks that are genuinely close to the query.
            if score < 0.45:
                sources.append({
                    "file": node.metadata.get("file_name", "Document"),
                    "excerpt": node.text[:400],
                    "score": round(score, 3)
                })
        return str(response)

    def web_search(query: str) -> str:
        """Search the web for current or external information not in the documents."""
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Web search is disabled. Add TAVILY_API_KEY to your .env file."
        from tavily import TavilyClient
        log.append(f"🌐 Web search: *{query}*")
        client = TavilyClient(api_key=api_key)
        result = client.search(query=query, max_results=3)
        snippets = [f"[{r['url']}]\n{r['content']}" for r in result.get("results", [])]
        return "\n\n".join(snippets) or "No results found."

    def calculate(expression: str) -> str:
        """Evaluate a math expression like '15 * 47' or '(100 + 50) / 3'."""
        log.append(f"🔢 Calculated: *{expression}*")
        try:
            return f"Result: {eval(expression)}"
        except Exception as e:
            return f"Error: {e}"

    def wikipedia_search(query: str) -> str:
        """Search Wikipedia for factual information about a topic, person, place, or concept."""
        import wikipedia
        log.append(f"📖 Wikipedia search: *{query}*")
        try:
            results = wikipedia.search(query, results=3)
            if not results:
                return f"No Wikipedia articles found for '{query}'"
            summary = wikipedia.summary(results[0], sentences=5, auto_suggest=False)
            return f"**{results[0]}** (Wikipedia)\n\n{summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                summary = wikipedia.summary(e.options[0], sentences=5, auto_suggest=False)
                return f"**{e.options[0]}** (Wikipedia)\n\n{summary}"
            except Exception:
                return f"Multiple matches found. Did you mean: {', '.join(e.options[:5])}?"
        except Exception as e:
            return f"Wikipedia error: {str(e)}"

    def dictionary_lookup(word: str) -> str:
        """Look up the definition, pronunciation, and usage examples for a word."""
        import requests
        log.append(f"📚 Dictionary lookup: *{word}*")
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word.strip().lower()}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return f"No definition found for '{word}'"
            data = resp.json()[0]
            lines = [f"**{data['word']}**"]
            if data.get("phonetic"):
                lines.append(f"Pronunciation: {data['phonetic']}")
            for meaning in data.get("meanings", [])[:2]:
                lines.append(f"\n*{meaning['partOfSpeech']}*")
                for defn in meaning.get("definitions", [])[:2]:
                    lines.append(f"• {defn['definition']}")
                    if defn.get("example"):
                        lines.append(f'  e.g. "{defn["example"]}"')
            return "\n".join(lines)
        except Exception as e:
            return f"Dictionary error: {str(e)}"

    def summarize(text: str) -> str:
        """Summarize a long piece of text into concise bullet points."""
        log.append(f"✍️ Summarizing text ({len(text.split())} words)...")
        response = Settings.llm.complete(
            f"Summarize the following text into 5 clear, concise bullet points:\n\n{text}"
        )
        return str(response)

    def translate(text: str, target_language: str) -> str:
        """Translate text into another language. Provide the text and the target language name."""
        log.append(f"🌍 Translating to *{target_language}*...")
        response = Settings.llm.complete(
            f"Translate the following text to {target_language}. "
            f"Only return the translated text, nothing else:\n\n{text}"
        )
        return str(response)

    def extract_keywords(text: str) -> str:
        """Extract the main keywords and key topics from a piece of text."""
        log.append(f"🔑 Extracting keywords from text ({len(text.split())} words)...")
        response = Settings.llm.complete(
            f"Extract the 10 most important keywords and topics from the following text. "
            f"Return them as a numbered list, each with a one-line explanation:\n\n{text}"
        )
        return str(response)

    return [
        FunctionTool.from_defaults(fn=search_documents, name="document_search",
            description="Search indexed documents. Use FIRST for any question about uploaded files or URLs."),
        FunctionTool.from_defaults(fn=web_search, name="web_search",
            description="Search the web for current events or info not in documents."),
        FunctionTool.from_defaults(fn=wikipedia_search, name="wikipedia_search",
            description="Search Wikipedia for factual info about topics, people, places, or concepts."),
        FunctionTool.from_defaults(fn=dictionary_lookup, name="dictionary_lookup",
            description="Look up a word's definition, pronunciation, and usage examples."),
        FunctionTool.from_defaults(fn=summarize, name="summarize",
            description="Summarize a long piece of text into bullet points."),
        FunctionTool.from_defaults(fn=translate, name="translate",
            description="Translate text into another language. Needs the text and target language name."),
        FunctionTool.from_defaults(fn=extract_keywords, name="extract_keywords",
            description="Extract the main keywords and topics from a piece of text."),
        FunctionTool.from_defaults(fn=calculate, name="calculator",
            description="Evaluate math expressions like '15 * 47'."),
    ]

def build_agent(query_engine, log: list, sources: list) -> ReActAgent:
    return ReActAgent(
        tools=make_tools(query_engine, log, sources),
        llm=Settings.llm,
        verbose=False,
        system_prompt=(
            "You are a research assistant. "
            "STRICT RULE: document_search MUST be your very first tool call for EVERY question — no exceptions. "
            "If document_search returns a useful answer, STOP. Do not call any other tool. "
            "Only consider other tools after document_search explicitly returns nothing useful.\n\n"
            "After document_search finds nothing, follow this priority:\n"
            "- wikipedia_search: for factual or encyclopaedic questions\n"
            "- web_search: for current events or very recent information\n"
            "- dictionary_lookup: ONLY when the user asks for a word definition\n"
            "- summarize: ONLY when the user explicitly asks to summarize something\n"
            "- translate: ONLY when the user explicitly asks to translate\n"
            "- extract_keywords: ONLY when the user explicitly asks for keywords\n"
            "- calculator: ONLY for math expressions\n\n"
            "If nothing is found anywhere, say so honestly."
        )
    )

async def ask(agent: ReActAgent, question: str) -> str:
    return str(await agent.run(question))

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_sources" not in st.session_state:
    st.session_state.indexed_sources = load_sources()  # reload from disk on startup

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Documents")

    uploaded = st.file_uploader(
        "Upload files",
        type=["txt", "pdf", "docx", "csv"],
        accept_multiple_files=True
    )
    url_input = st.text_input("Or paste a webpage URL", placeholder="https://...")

    if st.button("Index", type="primary"):
        docs, names, errors = [], [], []

        with st.status("Processing sources...", expanded=True) as status:
            for f in (uploaded or []):
                st.write(f"📄 Reading {f.name}...")
                try:
                    docs.append(parse_uploaded_file(f))
                    names.append(f.name)
                except Exception as e:
                    errors.append(f"{f.name}: {e}")

            if url_input.strip():
                st.write(f"🌐 Fetching {url_input.strip()}...")
                try:
                    docs.append(fetch_url(url_input.strip()))
                    names.append(url_input.strip())
                except Exception as e:
                    errors.append(f"URL error: {e}")

            if docs:
                st.write("🧠 Embedding and storing in ChromaDB...")
                add_to_index(docs)
                st.session_state.indexed_sources.extend(names)
                save_sources(st.session_state.indexed_sources)
                status.update(label=f"Indexed {len(docs)} source(s)!", state="complete")
            else:
                status.update(label="Nothing to index.", state="error")

        for err in errors:
            st.error(err)

    if st.session_state.indexed_sources:
        st.divider()
        st.subheader("Indexed sources")
        for name in st.session_state.indexed_sources:
            label = name if len(name) <= 45 else "..." + name[-42:]
            st.write(f"✅ {label}")

    count = get_chunk_count()
    if count > 0:
        st.caption(f"{count} chunks in ChromaDB — persists across restarts")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("💣 Reset DB"):
            reset_db()
            st.session_state.indexed_sources = []
            st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────
st.title("🔍 AI Research Assistant")
st.caption("Upload documents, add URLs, then ask questions — powered by RAG + Agents")

if get_chunk_count() == 0:
    st.info("👈 Add documents or a URL in the sidebar and click **Index** to get started.")

# Replay previous messages (with reasoning + sources preserved)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("tool_log"):
            with st.expander("🔧 Agent steps"):
                for entry in msg["tool_log"]:
                    st.markdown(entry)
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['file']}** — relevance score: {src['score']}")
                    st.caption(src["excerpt"])
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        log, sources = [], []

        with st.status("Working on it...", expanded=True) as status:
            st.write("🤔 Deciding which tools to use...")
            query_engine = get_index().as_query_engine(similarity_top_k=3)
            agent = build_agent(query_engine, log, sources)
            answer = asyncio.run(ask(agent, prompt))
            status.update(label="Done!", state="complete", expanded=False)

        st.write(answer)

        if log:
            with st.expander("🔧 Agent steps"):
                for entry in log:
                    st.markdown(entry)

        if sources:
            with st.expander(f"📚 Sources ({len(sources)})"):
                for src in sources:
                    st.markdown(f"**{src['file']}** — relevance score: {src['score']}")
                    st.caption(src["excerpt"])
                    st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "tool_log": log.copy(),
            "sources": sources.copy()
        })
