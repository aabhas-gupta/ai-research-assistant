from fastmcp import FastMCP

# Create a named MCP server
mcp = FastMCP("Research Tools")

@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluates a math expression like '15 * 47'"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
def word_count(text: str) -> str:
    """Counts the words in a given text"""
    count = len(text.split())
    return f"The text has {count} words."

if __name__ == "__main__":
    mcp.run()
