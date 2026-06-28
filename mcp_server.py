import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from groq import Groq as GroqClient

load_dotenv()

mcp = FastMCP("Research Tools")

# Groq client reused across LLM-powered tools
_groq = GroqClient(api_key=os.getenv("GROQ_API_KEY"))

def _llm(prompt: str) -> str:
    """Call the LLM for text-based tools."""
    response = _groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a math expression like '15 * 47' or '(100 + 50) / 3'."""
    try:
        return f"Result: {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
def word_count(text: str) -> str:
    """Count the number of words in a given text."""
    return f"The text has {len(text.split())} words."

@mcp.tool()
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for factual information about a topic, person, place, or concept."""
    import wikipedia
    try:
        results = wikipedia.search(query, results=3)
        if not results:
            return f"No Wikipedia articles found for '{query}'"
        summary = wikipedia.summary(results[0], sentences=5, auto_suggest=False)
        return f"{results[0]} (Wikipedia)\n\n{summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            summary = wikipedia.summary(e.options[0], sentences=5, auto_suggest=False)
            return f"{e.options[0]} (Wikipedia)\n\n{summary}"
        except Exception:
            return f"Multiple matches: {', '.join(e.options[:5])}"
    except Exception as e:
        return f"Wikipedia error: {str(e)}"

@mcp.tool()
def dictionary_lookup(word: str) -> str:
    """Look up the definition, pronunciation, and usage examples for a word."""
    import requests
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word.strip().lower()}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return f"No definition found for '{word}'"
        data = resp.json()[0]
        lines = [data["word"]]
        if data.get("phonetic"):
            lines.append(f"Pronunciation: {data['phonetic']}")
        for meaning in data.get("meanings", [])[:2]:
            lines.append(f"\n{meaning['partOfSpeech']}")
            for defn in meaning.get("definitions", [])[:2]:
                lines.append(f"• {defn['definition']}")
                if defn.get("example"):
                    lines.append(f'  e.g. "{defn["example"]}"')
        return "\n".join(lines)
    except Exception as e:
        return f"Dictionary error: {str(e)}"

@mcp.tool()
def summarize(text: str) -> str:
    """Summarize a long piece of text into concise bullet points."""
    return _llm(f"Summarize the following text into 5 clear, concise bullet points:\n\n{text}")

@mcp.tool()
def translate(text: str, target_language: str) -> str:
    """Translate text into another language. Provide the text and target language name."""
    return _llm(
        f"Translate the following text to {target_language}. "
        f"Only return the translated text, nothing else:\n\n{text}"
    )

@mcp.tool()
def extract_keywords(text: str) -> str:
    """Extract the main keywords and key topics from a piece of text."""
    return _llm(
        f"Extract the 10 most important keywords and topics from the following text. "
        f"Return them as a numbered list, each with a one-line explanation:\n\n{text}"
    )

if __name__ == "__main__":
    mcp.run()
