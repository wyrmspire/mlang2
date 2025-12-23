
import requests
import time
import sys
import json

BASE_URL = "http://localhost:8000"

def check_backend():
    global BASE_URL
    for port in [8000, 8001]:
        try:
            requests.get(f"http://localhost:{port}/health", timeout=1)
            BASE_URL = f"http://localhost:{port}"
            print(f"Detected backend on {BASE_URL}")
            return True
        except:
            pass
    return False

def test_start_live_via_lab():
    print("Testing 'Start live mode' via Lab Agent...")
    
    # 1. Send the chat command
    payload = {
        "messages": [
            {"role": "user", "content": "Start live mode for MES"}
        ]
    }
    
    try:
        resp = requests.post(f"{BASE_URL}/lab/agent", json=payload)
        resp.raise_for_status()
        data = resp.json()
        print("Response received:")
        print(json.dumps(data, indent=2))
        
        reply = data.get("reply", "")
        
        # Check if it claims to have started
        if "Live Simulation Started" in reply:
            print("Agent claims simulation started.")
        else:
            print("Agent did not start simulation.")
            return

        # 2. Check if we can find this session in /replay/sessions
        sessions_resp = requests.get(f"{BASE_URL}/replay/sessions")
        sessions = sessions_resp.json().get("sessions", [])
        
        print("\nActive Sessions in /replay/sessions:")
        print(sessions)
        
        if len(sessions) == 0:
            print("\n❌ CRITICAL ISSUE: The Lab Agent started a process but it is NOT registered in the Replay Sessions.")
            print("The user cannot view this simulation because the backend doesn't know it exists for streaming.")
        else:
            print(f"\n✅ Found {len(sessions)} sessions!")
            
    except Exception as e:
        print(f"Error testing: {e}")

if __name__ == "__main__":
    # Ensure server is up (basic check)
    if not check_backend():
        print("Server not running on port 8000 or 8001. Please start server first.")
        sys.exit(1)
        
    test_start_live_via_lab()
