# VizDoom Dynamic Difficulty Dashboard

This dashboard provides a live interface for monitoring features extracted from gameplay, LLM inputs and outputs, and the current state of the dynamic difficulty adjustment system.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the API server:
   ```bash
   python run_api.py
   ```

3. In another terminal, run the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

## API Endpoints

The API server provides the following endpoints for integration:

- `GET /data`: Retrieve current dashboard data
- `POST /features`: Update features (expects JSON: `{"features": {...}}`)
- `POST /llm_input`: Update LLM input (expects JSON: `{"input_text": "..."}`)
- `POST /llm_output`: Update LLM output (expects JSON: `{"output_text": "..."}`)
- `POST /state`: Update current state (expects JSON: `{"state": "..."}`)

## Integration

The feature extraction script and LLM component should POST updates to the respective endpoints to keep the dashboard live.

Example Python code to update features:
```python
import requests

features = {"health": 80, "ammo": 50, "kills": 10}
requests.post("http://localhost:8000/features", json={"features": features})
```

## Testing

1. Start the API server in one terminal:
   ```bash
   python run_api.py
   ```

2. In another terminal, run the test script to populate data:
   ```bash
   python test_api.py
   ```

3. In a third terminal, start the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

The dashboard will open in your browser and update every 2 seconds. You can run `test_api.py` multiple times to see live updates.

You can also manually test the API:
```bash
curl http://localhost:8000/data
curl -X POST http://localhost:8000/state -H "Content-Type: application/json" -d '{"state": "Testing..."}'
```