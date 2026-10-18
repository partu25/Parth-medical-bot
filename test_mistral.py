import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")

print(f"==================================================")
print(f"Testing Mistral API Key: {API_KEY[:10]}...{API_KEY[-6:]}")
print(f"==================================================\n")

# Model to use on this tier
MODEL_NAME = "open-mistral-nemo"

# --- Test 1: Direct HTTP API Call ---
print(f"--- [Test 1] Direct REST API Call ({MODEL_NAME}) ---")
url = "https://api.mistral.ai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
payload = {
    "model": MODEL_NAME,
    "messages": [
        {"role": "user", "content": "Respond in 5 words: Is this API key working?"}
    ],
    "max_tokens": 50
}

try:
    response = requests.post(url, headers=headers, json=payload, timeout=15)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("✅ Direct API Success!")
        print(f"Response: {data['choices'][0]['message']['content'].strip()}\n")
    else:
        print("❌ Direct API Error:")
        print(response.text, "\n")
except Exception as e:
    print(f"❌ Connection Error: {e}\n")

# --- Test 2: LangChain ChatMistralAI Integration ---
print(f"--- [Test 2] LangChain ChatMistralAI Integration ({MODEL_NAME}) ---")
try:
    llm = ChatMistralAI(model=MODEL_NAME, api_key=API_KEY)
    res = llm.invoke("What are 2 early signs of diabetes? Answer in one short sentence.")
    print("✅ ChatMistralAI Success!")
    print(f"Response: {res.content.strip()}\n")
except Exception as e:
    print(f"❌ ChatMistralAI Error: {e}\n")

print("==================================================")
print("Test completed!")
print("==================================================")
