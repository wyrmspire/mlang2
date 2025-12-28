#!/usr/bin/env python
"""
Terminal chat interface for agents.

Usage:
    python scripts/agent_chat.py --agent lab     # Agent 2: Brainstormer (light mode, fast)
    python scripts/agent_chat.py --agent tradeviz  # Agent 1: TradeViz (full mode, creates viz)
"""
import argparse
import requests
import json
import sys


def chat_lab(message: str, history: list) -> str:
    """Chat with Lab Agent (Agent 2 - Brainstormer)."""
    messages = history + [{"role": "user", "content": message}]
    
    resp = requests.post(
        "http://localhost:8000/lab/agent",
        json={"messages": messages},
        timeout=120
    )
    resp.raise_for_status()
    return resp.json().get("reply", "No response")


def chat_tradeviz(message: str, history: list) -> str:
    """Chat with TradeViz Agent (Agent 1 - Full mode)."""
    messages = history + [{"role": "user", "content": message}]
    
    resp = requests.post(
        "http://localhost:8000/agent/chat",
        json={
            "messages": messages,
            "context": {
                "runId": "",
                "currentIndex": 0,
                "currentMode": "exploration"
            }
        },
        timeout=120
    )
    resp.raise_for_status()
    return resp.json().get("reply", "No response")


def main():
    parser = argparse.ArgumentParser(description="Terminal chat with agents")
    parser.add_argument("--agent", choices=["lab", "tradeviz"], default="lab",
                        help="Which agent to chat with")
    args = parser.parse_args()
    
    chat_fn = chat_lab if args.agent == "lab" else chat_tradeviz
    agent_name = "Lab (Brainstormer)" if args.agent == "lab" else "TradeViz (Full)"
    
    print(f"\n[AGENT: {agent_name}]")
    print("=" * 50)
    print("Type your message. Press Ctrl+C to exit.\n")
    
    history = []
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            
            print("Agent: Thinking...")
            response = chat_fn(user_input, history)
            
            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            
            print(f"\nAgent:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
