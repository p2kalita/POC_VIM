"""
Example application using vLLM endpoint
Shows how to integrate into your own project
"""

from openai import OpenAI

# ============================================
# 👇 REPLACE WITH YOUR NGROK URL FROM COLAB
# ============================================
PUBLIC_URL = "https://dried-moonscape-violet.ngrok-free.dev"
# ============================================


class LLMClient:
    """Wrapper around vLLM endpoint"""
    
    def __init__(self, base_url):
        self.client = OpenAI(
            api_key="dummy-key",
            base_url=f"{base_url}/v1"
        )
    
    def chat(self, message, max_tokens=150):
        """Send a chat message and get response"""
        response = self.client.chat.completions.create(
            # model="Qwen/Qwen2.5-3B-Instruct",
            model="/content/models/Qwen2.5-3B-Instruct",
            messages=[{"role": "user", "content": message}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def complete(self, prompt, max_tokens=150):
        """Complete a text prompt"""
        response = self.client.completions.create(
            model="default",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].text
    
    def summarize(self, text):
        """Summarize text"""
        prompt = f"Summarize this text in 2-3 sentences:\n\n{text}"
        return self.chat(prompt, max_tokens=150)
    
    def translate(self, text, target_lang):
        """Translate text to another language"""
        prompt = f"Translate this text to {target_lang}:\n\n{text}"
        return self.chat(prompt)


# Example usage
if __name__ == "__main__":
    if "YOUR_NGROK_URL" in PUBLIC_URL:
        print("❌ Update PUBLIC_URL first!")
        exit(1)
    
    print("🚀 Initializing LLM client...")
    llm = LLMClient(PUBLIC_URL)
    
    # Example 1: Simple chat
    print("\n" + "="*60)
    print("Example 1: Chat")
    print("="*60)
    response = llm.chat("Tell me a funny joke about programmers")
    print(f"Response: {response}")
    
    # Example 2: Text completion
    print("\n" + "="*60)
    print("Example 2: Text Completion")
    print("="*60)
    prompt = "The capital of France is"
    response = llm.complete(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    # Example 3: Summarization
    print("\n" + "="*60)
    print("Example 3: Summarization")
    print("="*60)
    text = """
    Artificial Intelligence is transforming the way we work and live.
    Machine learning models can now recognize images, understand language,
    and even generate creative content. This technology is being applied
    across industries from healthcare to finance to entertainment.
    However, it also raises important questions about privacy, bias,
    and job displacement that society must address.
    """
    summary = llm.summarize(text)
    print(f"Original text length: {len(text)} chars")
    print(f"Summary: {summary}")
    
    # Example 4: Translation
    print("\n" + "="*60)
    print("Example 4: Translation")
    print("="*60)
    english_text = "Hello, how are you today?"
    spanish = llm.translate(english_text, "Spanish")
    print(f"English: {english_text}")
    print(f"Spanish: {spanish}")
    
    print("\n✅ All examples completed!\n")