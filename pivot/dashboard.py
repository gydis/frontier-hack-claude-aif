import streamlit as st
import requests
import pandas as pd
import numpy as np
from src.discretizer import discretize_stats  # Import your discretizer

# Page Configuration
st.set_page_config(page_title="VizDoom Live Analytics", layout="wide")

st.title("🎮 VizDoom Dynamic Difficulty Dashboard")
st.markdown("---")

API_URL = "http://localhost:8000/data"

# --- 1. INITIALIZE SESSION STATE ---
if "accuracy_history" not in st.session_state:
    st.session_state.accuracy_history = []
if "max_history_points" not in st.session_state:
    st.session_state.max_history_points = 100


def get_status_color(label):
    """Helper to add some visual flair to labels."""
    if label == "poor":
        return f"🔴 {label.upper()}"
    if label == "high":
        return f"🟢 {label.upper()}"
    return f"🔵 {label.upper()}"


@st.fragment(run_every=1)
def update_dashboard():
    try:
        response = requests.get(API_URL, timeout=0.5)
        if response.status_code == 200:
            data = response.json()
            features = data.get("features", {})
            current_acc = features.get("accuracy", 0)

            # --- 2. UPDATE HISTORY ---
            st.session_state.accuracy_history.append(current_acc)
            if len(st.session_state.accuracy_history) > st.session_state.max_history_points:
                st.session_state.accuracy_history.pop(0)

            # --- 3. CALCULATE LIVE MEAN ---
            history_array = st.session_state.accuracy_history
            live_mean = sum(history_array) / \
                len(history_array) if history_array else 0

            # --- 4. DISPLAY STATISTICS ---
            st.header("📊 Tracked Player Statistics")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Current Accuracy", f"{current_acc:.1%}")
            with m2:
                st.metric("Rolling Avg Accuracy", f"{live_mean:.1%}")
            with m3:
                st.metric("Data Points Collected", len(history_array))

            # --- 5. LIVE GRAPH ---
            st.subheader("Accuracy Trend (Live)")
            if len(history_array) > 1:
                chart_data = pd.DataFrame(history_array, columns=["Accuracy"])
                st.line_chart(chart_data, height=250)
            else:
                st.info("Collecting data points to build trend line...")

            st.markdown("---")

            # --- 6. DISCRETIZED STATUS ---
            st.header("🧠 Discretized Player Status")
            # Run the discretizer on current features
            discrete_data = discretize_stats(features)

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.write("**Accuracy**")
                st.markdown(get_status_color(discrete_data.get("accuracy")))
            with d2:
                st.write("**KDR**")
                st.markdown(get_status_color(discrete_data.get("kdr")))
            with d3:
                st.write("**Frags/Min**")
                st.markdown(get_status_color(
                    discrete_data.get("frags_per_minute")))
            with d4:
                st.write("**Damage Ratio**")
                st.markdown(get_status_color(
                    discrete_data.get("damage_ratio")))

            st.markdown("---")

            # --- 7. AGENT CONFIGS ---
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
