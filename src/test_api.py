import requests
import subprocess
import time
import sys

API_URL = "http://127.0.0.1:8000/predict"
API_HOST = "127.0.0.1"
API_PORT = 8000

def is_api_running():
    try:
        r = requests.get(f"http://{API_HOST}:{API_PORT}/")
        return True
    except requests.exceptions.ConnectionError:
        return False

# Start API if not running
if not is_api_running():
    print("⚡ API not running. Starting Flask API...")
    # Start Flask server in a subprocess
    api_process = subprocess.Popen(
        [sys.executable, "src/api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    # Wait a few seconds for API to start
    time.sleep(5)
else:
    api_process = None
    print("✅ API is already running.")

# Now test the API
emails_to_test = [
    "Congratulations! You won free cash. Claim now.",
    "Hey, are we meeting tomorrow for lunch?",
    "Win a brand new iPhone now!",
    "Please review the attached document."
]

for email_text in emails_to_test:
    data = {"email": email_text}
    try:
        response = requests.post(API_URL, json=data)
        print(f"Email: {email_text}\nPrediction: {response.json()['prediction']}\n")
    except requests.exceptions.ConnectionError:
        print("❌ Failed to connect to the API. Is it running?")
        break

# Optional: terminate the Flask server if we started it
if api_process:
    print("Shutting down API...")
    api_process.terminate()