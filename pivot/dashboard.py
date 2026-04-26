import streamlit as st
import requests
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(page_title="VizDoom Live Analytics", layout="wide")

st.title("🎮 VizDoom Dynamic Difficulty Dashboard")
st.markdown("---")

API_URL = "http://localhost:8000/data"

# --- 1. INITIALIZE SESSION STATE ---
# This keeps track of history across reruns
if "accuracy_history" not in st.session_state:
    st.session_state.accuracy_history = []
if "max_history_points" not in st.session_state:
    st.session_state.max_history_points = 100  # Keep last 100 seconds of data


@st.fragment(run_every=1)  # Updated to 1s for smoother "live" feel
def update_dashboard():
    try:
        response = requests.get(API_URL, timeout=0.5)
        if response.status_code == 200:
            data = response.json()
            features = data.get("features", {})
            current_acc = features.get("accuracy", 0)

            # --- 2. UPDATE HISTORY ---
            st.session_state.accuracy_history.append(current_acc)

            # Prevent the list from growing infinitely (memory management)
            if len(st.session_state.accuracy_history) > st.session_state.max_history_points:
                st.session_state.accuracy_history.pop(0)

            # --- 3. CALCULATE LIVE MEAN ---
            # Using standard math:
            # $\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$
            history_array = st.session_state.accuracy_history
            live_mean = sum(history_array) / \
                len(history_array) if history_array else 0

            # --- 4. DISPLAY STATISTICS ---
            st.header("📊 Tracked Player Statistics")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Current Accuracy", f"{current_acc:.1%}")
            with m2:
                # Show the live mean calculation
                st.metric("Rolling Avg Accuracy", f"{live_mean:.1%}")
            with m3:
                st.metric("Data Points Collected", len(history_array))

            # --- 5. LIVE GRAPH ---
            st.subheader("Accuracy Trend (Live)")
            if len(history_array) > 1:
                # Convert history to a DataFrame for Streamlit's chart
                chart_data = pd.DataFrame(history_array, columns=["Accuracy"])
                st.line_chart(chart_data, height=250)
            else:
                st.info("Collecting data points to build trend line...")

            st.markdown("---")

            # --- 6. AGENT CONFIGS ---
            st.header("⚙️ Agent Updated Configs")
            agent_decision = data.get("llm_output", "{}")
            st.code(agent_decision, language="json")

            st.caption(f"Last updated: {data.get('last_updated', 'N/A')}")

        else:
            st.error("API reachable but returned an error.")
    except Exception as e:
        st.error(f"Connection Error: {e}")


# Call the fragment
update_dashboard()

# Sidebar controls
with st.sidebar:
    st.header("Dashboard Settings")
    if st.button("Clear History"):
        st.session_state.accuracy_history = []
        st.rerun()
    st.session_state.max_history_points = st.slider(
        "History Buffer (seconds)", 10, 500, 100)
