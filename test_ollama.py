import ollama

response = ollama.chat(
    model="phi3:mini",
    messages=[
        {
            "role": "user",
            "content": "Say hello"
        }
    ]
)

print(response["message"]["content"])