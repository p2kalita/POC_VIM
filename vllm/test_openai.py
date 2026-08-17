from openai import OpenAI

# UPDATE THIS WITH YOUR NGROK URL FROM COLAB
PUBLIC_URL = "https://dried-moonscape-violet.ngrok-free.dev"

client = OpenAI(
    api_key="dummy-key",
    base_url=f"{PUBLIC_URL}/v1"
)

print("🔄 Sending chat request...")

response = client.chat.completions.create(
    # model="Qwen/Qwen2.5-3B-Instruct",
    model="/content/models/Qwen2.5-3B-Instruct",
    messages=[
        {"role": "user", "content": "What is 2+2?"}
    ],
    max_tokens=50
)

print("✅ Response received!")
print(f"Answer: {response.choices[0].message.content}")

