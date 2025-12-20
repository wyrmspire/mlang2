import requests
import json
import time
import sys

# Try both ports
PORTS = [8000, 8001]

def test_replay():
    base_url = None
    for port in PORTS:
        try:
            resp = requests.get(f"http://localhost:{port}/health")
            if resp.status_code == 200:
                base_url = f"http://localhost:{port}"
                break
        except:
            pass
            
    if not base_url:
        print("Could not find running backend on 8000 or 8001")
        sys.exit(1)
        
    print(f"Connected to backend at {base_url}")

    # 1. Start Replay
    print("Starting replay...")
    # Use fallback date we put in UI
    start_date = "2025-03-18T09:30:00"
    
    payload = {
        "model_path": "models/swing_breakout_model.pth",
        "start_date": start_date,
        "days": 1,
        "speed": 100.0
    }
    
    try:
        resp = requests.post(f"{base_url}/replay/start", json=payload)
        print("Start Resp:", resp.status_code, resp.text)
        if resp.status_code != 200:
            return
            
        data = resp.json()
        sid = data['session_id']
        print(f"Session ID: {sid}")
    except Exception as e:
        print(f"Start failed: {e}")
        return
    
    # 2. Consume Stream
    print(f"Connecting to stream {base_url}/replay/stream/{sid}...")
    try:
        with requests.get(f"{base_url}/replay/stream/{sid}", stream=True, timeout=10) as stream:
            print(f"Stream Status: {stream.status_code}")
            if stream.status_code != 200:
                print("Stream failed")
                return
                
            print("Stream connected. Reading lines...")
            count = 0
            for line in stream.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    print(f"Received: {decoded[:100]}...") # Truncate
                    count += 1
                    if count >= 10:
                        print("Got 10 events. Success.")
                        break
    except Exception as e:
        print(f"Stream error: {e}")
    finally:
        # Cleanup
        print("Stopping session...")
        requests.delete(f"{base_url}/replay/sessions/{sid}")

if __name__ == "__main__":
    test_replay()
