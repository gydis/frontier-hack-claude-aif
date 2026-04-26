from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
from datetime import datetime

app = FastAPI()

DATA_FILE = "dashboard_data.json"

# Initialize data structure
if not os.path.exists(DATA_FILE):
    initial_data = {
        "features": {},
        "llm_input": "",
        "llm_output": "",
        "current_state": "Initializing...",
        "last_updated": str(datetime.now())
    }
    with open(DATA_FILE, "w") as f:
        json.dump(initial_data, f)


class FeaturesData(BaseModel):
    features: dict


class LLMInputData(BaseModel):
    input_text: str


class LLMOutputData(BaseModel):
    output_text: str


class StateData(BaseModel):
    state: str


@app.get("/data")
def get_data():
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    return data


@app.post("/features")
def update_features(data: FeaturesData):
    with open(DATA_FILE, "r+") as f:
        current = json.load(f)
        current["features"] = data.features
        current["last_updated"] = str(datetime.now())
        f.seek(0)
        json.dump(current, f)
        f.truncate()
    return {"message": "Features updated"}


@app.post("/llm_input")
def update_llm_input(data: LLMInputData):
    with open(DATA_FILE, "r+") as f:
        current = json.load(f)
        current["llm_input"] = data.input_text
        current["last_updated"] = str(datetime.now())
        f.seek(0)
        json.dump(current, f)
        f.truncate()
    return {"message": "LLM input updated"}


@app.post("/llm_output")
def update_llm_output(data: LLMOutputData):
    with open(DATA_FILE, "r+") as f:
        current = json.load(f)
        current["llm_output"] = data.output_text
        current["last_updated"] = str(datetime.now())
        f.seek(0)
        json.dump(current, f)
        f.truncate()
    return {"message": "LLM output updated"}


@app.post("/state")
def update_state(data: StateData):
    with open(DATA_FILE, "r+") as f:
        current = json.load(f)
        current["current_state"] = data.state
        current["last_updated"] = str(datetime.now())
        f.seek(0)
        json.dump(current, f)
        f.truncate()
    return {"message": "State updated"}
