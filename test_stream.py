import ollama

stream = ollama.chat(
    model="gemma3:4b",
    messages=[
        {"role": "user", "content": "Hello"}
    ],
    stream=True,
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)