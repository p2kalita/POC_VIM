import requests

# UPDATE THIS WITH YOUR NGROK URL FROM COLAB
PUBLIC_URL = "https://dried-moonscape-violet.ngrok-free.dev"

url = f"{PUBLIC_URL}/v1/chat/completions"

payload = {
    # model="Qwen/Qwen2.5-3B-Instruct",
    model="/content/models/Qwen2.5-3B-Instruct",
    "prompt": "Hello, my name is",
    "max_tokens": 50,
    "temperature": 0.7
}

print("🔄 Sending request...")
response = requests.post(url, json=payload)

if response.status_code == 200:
    result = response.json()
    print("✅ Response received!")
    print(f"Text: {result['choices'][0]['text']}")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)