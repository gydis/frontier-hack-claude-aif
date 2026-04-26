# ViZDoom LLM-Driven Dynamic Difficulty Adjustment (DDA)

This project demonstrates a dynamic difficulty adjustment system within a ViZDoom deathmatch environment. It uses a LLM agent as a high-level controller to tune bot difficulty based on human performance.

## Overview
- **Core Loop:** A human plays deathmatch rounds against bots. After each round, performance stats (frags, deaths, accuracy, damage) are discretized and sent to the agent.
- **LLM Controller:** The agent analyzes the player's current performance and historical difficulty trends to select a new difficulty level, providing a natural language explanation for its choice.
- **Live Dashboard:** A Streamlit interface visualizes performance metrics, difficulty adjustment and the agent's reasoning in real-time.
- **Control Conditions:** Supports fixed difficulty and rule-based adjustment for comparative analysis.

## Project Structure
- `pivot/api`: FastAPI server handling LLM logic.
- `pivot/dashboard.py`: Streamlit visualization tool.
- `scripts/run_llm_session.py`: Main game loop and ViZDoom environment manager.
- `data_dashboard.json`: Local log file where all session results are stored.

## Setup

Before running the application, create a `.env` file in the **root directory** of the project and add your Anthropic API key:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

## Execution Instructions

Run each command in a separate terminal window:

**Terminal 1: API Server**
```bash
uvicorn pivot.api:app --reload
```
**Terminal 2: Visualization Dashboard**
```bash
streamlit run pivot/dashboard.py
```

Terminal 3: Game Session
```bash
Bash
python scripts/run_llm_session.py
```

Logging

All session data, including player statistics and LLM decision history, are logged to `data_dashboard.json` in the root folder for post-game analysis.