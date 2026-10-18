import os
import requests
from dotenv import load_dotenv

load_dotenv()
account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
api_token = os.getenv("CLOUDFLARE_API_TOKEN")
model = "@cf/baai/bge-small-en-v1.5"

url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": "application/json"
}
payload = {
    "text": ["What are early symptoms of Type 2 Diabetes?"]
}

print(f"Testing Cloudflare Workers AI Embedding model: {model}...")
try:
    response = requests.post(url, headers=headers, json=payload, timeout=15)
    print("Status code:", response.status_code)
    data = response.json()
    if response.status_code == 200 and data.get("success"):
        res = data.get("result", {})
        data_arr = res.get("data", [])
        if data_arr:
            dim = len(data_arr[0])
            print(f"✅ Success! Embedding generated successfully.")
            print(f"Vector dimension: {dim} (matches BAAI/bge-small-en-v1.5)")
            print(f"Sample values: {data_arr[0][:5]}...")
        else:
            print("Response:", res)
    else:
        print("❌ Cloudflare Error:", data)
except Exception as e:
    print("❌ Exception:", e)
