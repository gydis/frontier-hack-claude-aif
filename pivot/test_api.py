#!/usr/bin/env python3
import requests
import time
import random

API_BASE = "http://localhost:8000"


def test_api():
    print("Testing API endpoints...")

    # Test initial data
    response = requests.get(f"{API_BASE}/data")
    print(f"Initial data: {response.json()}")

    # Update features
    features = {
        "health": random.randint(0, 100),
        "ammo": random.randint(0, 50),
        "kills": random.randint(0, 20),
        "position": {"x": random.uniform(-10, 10), "y": random.uniform(-10, 10)}
    }
    requests.post(f"{API_BASE}/features", json={"features": features})
    print(f"Updated features: {features}")

    # Update LLM input
    llm_input = "Player health is low, ammo is depleting, suggest increasing enemy spawn rate."
    requests.post(f"{API_BASE}/llm_input", json={"input_text": llm_input})
    print(f"Updated LLM input: {llm_input}")

    # Update LLM output
    llm_output = "Increase enemy difficulty by 20%, spawn rate +15%, reduce health pickups."
    requests.post(f"{API_BASE}/llm_output", json={"output_text": llm_output})
    print(f"Updated LLM output: {llm_output}")

    # Update state
    state = "Adjusting difficulty based on LLM recommendation..."
    requests.post(f"{API_BASE}/state", json={"state": state})
    print(f"Updated state: {state}")

    # Check final data
    time.sleep(1)
    response = requests.get(f"{API_BASE}/data")
    print(f"Final data: {response.json()}")


if __name__ == "__main__":
    test_api()
