import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account

creds = service_account.Credentials.from_service_account_file(
    "gcp-service-account.json",
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
creds.refresh(Request())
token = creds.token

url = "https://us-central1-aiplatform.googleapis.com/v1/projects/agent-hackathon-463002/locations/us-central1/reasoningEngines/1135162192795009024:run"

payload = {
    "input": {}  # ❗如果 agent 本身不需要 external input
}

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

try:
    print("✅ Status:", response.status_code)
    print("✅ Output:", response.json())
except Exception:
    print("❌ Failed to parse JSON. Raw response:")
    print(response.text)
