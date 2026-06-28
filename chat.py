import os
from dotenv import load_dotenv
from groq import Groq

# Load the API key from .env file
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Chat history — Groq needs us to track this manually
# Each message is a dict with "role" and "content"
history = [
    {"role": "system", "content": "You are a helpful research assistant."}
]

print("AI Assistant ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    # Add user message to history
    history.append({"role": "user", "content": user_input})

    # Send full history to the model
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=history,
    )

    # Extract the reply
    reply = response.choices[0].message.content

    # Add AI reply to history so it remembers context
    history.append({"role": "assistant", "content": reply})

    print(f"AI: {reply}\n")
