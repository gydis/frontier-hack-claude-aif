### Current Game State
- Episode Number: {{episode_count}}
- Current Difficulty Level: {{current_level}}/5

### Player Performance (Discretized)
- Accuracy: {{labels.accuracy}}
- Frags Per Minute: {{labels.fpm}}
- K/D Ratio: {{labels.kd}}
- Damage Taken: {{labels.damage}}

### Decision History (Last 3 Episodes)
{{history_summary}}

### Instructions
1. Analyze if the player is bored (over-performing) or frustrated (under-performing).
2. Select a Difficulty Level (1-5). You may only change the level by +/- 1 from the current state.
3. Map that level to specific values for the ViZDoom knobs.
4. Provide a one-sentence reasoning for the dashboard explaining why you adjusted those specific knobs.

### Expected JSON Schema
{
  "next_difficulty_level": int,
  "cfg_updates": {
    "doom_skill": int,
    "vizdoom_bot_count": int,
    "FastMonsters": int,
    "sv_itemrespawnrate": int
  },
  "reasoning": "string"
}