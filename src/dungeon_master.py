"""
DungeonMasterAgent — Active Inference agent that selects difficulty adjustments.

Uses pymdp v1.0.0 (JAX-based) with the following generative model:

Hidden state factors:
  Factor 0 — Engagement:  FRUSTRATED=0, FLOW=1, BORED=2
  Factor 1 — Difficulty:  skill 1..5  (indices 0..4)

Observation modalities:
  Modality 0 — Performance: LOW=0, MED=1, HIGH=2
  Modality 1 — Trend:       DECLINING=0, STABLE=1, IMPROVING=2

Actions (one per factor; only Factor 1 / Difficulty is controllable):
  LOWER=0, MAINTAIN=1, RAISE=2, ADD_BOT=3
"""

import jax
import jax.numpy as jnp
import numpy as np
from pymdp.agent import Agent

# State / obs dimensions
N_ENG   = 3   # FRUSTRATED, FLOW, BORED
N_DIFF  = 5   # skill 1..5
N_PERF  = 3   # LOW, MED, HIGH
N_TREND = 3   # DECLINING, STABLE, IMPROVING
N_ACT   = 4   # LOWER, MAINTAIN, RAISE, ADD_BOT

# Action indices
LOWER   = 0
MAINTAIN = 1
RAISE   = 2
ADD_BOT = 3

# Observation indices (re-exported for callers)
PERF_LOW,  PERF_MED,  PERF_HIGH   = 0, 1, 2
TREND_DEC, TREND_STA, TREND_IMP   = 0, 1, 2

# Engagement state indices
FRUSTRATED, FLOW, BORED = 0, 1, 2


def _build_A() -> list[jnp.ndarray]:
    """
    A[m] has shape (batch=1, n_obs_m, N_ENG, N_DIFF).
    P(obs | engagement, difficulty).
    Difficulty factor does not change the observation distribution —
    engagement state alone drives expected performance.
    """
    # Modality 0: Performance | Engagement
    #   FRUSTRATED → mostly LOW; FLOW → mostly MED; BORED → mostly HIGH
    perf_given_eng = np.array([
        [0.70, 0.15, 0.05],   # P(LOW  | FRUS, FLOW, BORED)
        [0.20, 0.70, 0.20],   # P(MED  | ...)
        [0.10, 0.15, 0.75],   # P(HIGH | ...)
    ], dtype=np.float32)  # shape (3, 3): [obs, engagement]

    A0 = np.zeros((1, N_PERF, N_ENG, N_DIFF), dtype=np.float32)
    for d in range(N_DIFF):
        A0[0, :, :, d] = perf_given_eng
    # Normalise: axis=1 (obs dimension) must sum to 1 for each (eng,diff)
    A0 /= A0.sum(axis=1, keepdims=True)

    # Modality 1: Trend — nearly uniform (learned online)
    A1 = np.ones((1, N_TREND, N_ENG, N_DIFF), dtype=np.float32) / N_TREND

    return [jnp.array(A0), jnp.array(A1)]


def _build_pA() -> list[jnp.ndarray]:
    """Dirichlet prior concentration for online A learning. Higher = slower learning."""
    pA0 = np.ones((1, N_PERF, N_ENG, N_DIFF), dtype=np.float32) * 2.0
    pA1 = np.ones((1, N_TREND, N_ENG, N_DIFF), dtype=np.float32) * 1.0
    return [jnp.array(pA0), jnp.array(pA1)]


def _build_B() -> list[jnp.ndarray]:
    """
    B[f] shape (batch=1, n_states_f, n_states_f, n_actions).
    P(state' | state, action).

    Factor 0 (Engagement): probabilistic — action affects engagement.
    Factor 1 (Difficulty): near-deterministic — DM controls it directly.
    """
    # --- Factor 0: Engagement transitions ---
    # B0[:, :, :, action] = P(eng' | eng, action)
    # Columns = current state, rows = next state
    B0 = np.zeros((1, N_ENG, N_ENG, N_ACT), dtype=np.float32)

    # LOWER difficulty
    # Frustrated → likely recover; Flow → might get bored; Bored → stays bored
    B0[0, :, :, LOWER] = np.array([
        [0.50, 0.20, 0.05],   # FRUSTRATED' | FRUS, FLOW, BORED
        [0.40, 0.60, 0.25],   # FLOW'
        [0.10, 0.20, 0.70],   # BORED'
    ])

    # MAINTAIN — mostly stays same, slight regression toward mean
    B0[0, :, :, MAINTAIN] = np.array([
        [0.70, 0.10, 0.05],
        [0.20, 0.80, 0.15],
        [0.10, 0.10, 0.80],
    ])

    # RAISE difficulty
    # Bored → likely finds flow; Flow → might get frustrated; Frustrated → worse
    B0[0, :, :, RAISE] = np.array([
        [0.70, 0.20, 0.05],
        [0.25, 0.55, 0.35],
        [0.05, 0.25, 0.60],
    ])

    # ADD_BOT — like a mild RAISE
    B0[0, :, :, ADD_BOT] = np.array([
        [0.65, 0.15, 0.05],
        [0.25, 0.65, 0.30],
        [0.10, 0.20, 0.65],
    ])

    # Normalise columns (axis=1 = next-state dim)
    B0 /= B0.sum(axis=1, keepdims=True)

    # --- Factor 1: Difficulty transitions (near-deterministic) ---
    B1 = np.zeros((1, N_DIFF, N_DIFF, N_ACT), dtype=np.float32)
    for a in range(N_ACT):
        for d in range(N_DIFF):
            if a == LOWER:
                nd = max(0, d - 1)
            elif a == RAISE or a == ADD_BOT:
                nd = min(N_DIFF - 1, d + 1)
            else:  # MAINTAIN
                nd = d
            B1[0, nd, d, a] = 0.95
            # Small probability of staying where we are (execution uncertainty)
            B1[0, d, d, a] += 0.05
    B1 /= B1.sum(axis=1, keepdims=True)

    return [jnp.array(B0), jnp.array(B1)]


def _build_C() -> list[jnp.ndarray]:
    """
    Log-preference over observations. Higher = more preferred.
    We want MED performance (flow) and STABLE/IMPROVING trend.
    """
    C0 = jnp.array([[-2.0, 4.0, -1.5]])   # (1, N_PERF)  LOW, MED, HIGH
    C1 = jnp.array([[-1.5, 2.0,  1.0]])   # (1, N_TREND) DEC, STA, IMP
    return [C0, C1]


def _build_D(initial_skill: int = 3) -> list[jnp.ndarray]:
    """Initial prior over hidden states."""
    D0 = jnp.array([[1 / N_ENG] * N_ENG])                     # (1, N_ENG) uniform
    skill_idx = max(0, min(N_DIFF - 1, initial_skill - 1))
    D1 = jnp.zeros((1, N_DIFF))
    D1 = D1.at[0, skill_idx].set(1.0)
    return [D0, D1]


class DungeonMasterAgent:
    """
    Wraps pymdp v1.0.0 Agent for episodic difficulty adjustment.

    Usage per episode:
        action_idx = dm.step(perf_obs, trend_obs)

    Returns one of: LOWER=0, MAINTAIN=1, RAISE=2, ADD_BOT=3
    """

    def __init__(self, initial_skill: int = 3, seed: int = 42):
        self._rng = jax.random.PRNGKey(seed)

        A   = _build_A()
        pA  = _build_pA()
        B   = _build_B()
        C   = _build_C()
        D   = _build_D(initial_skill)

        self._agent = Agent(
            A=A, B=B, C=C, D=D, pA=pA,
            policy_len=1,
            batch_size=1,
            action_selection="stochastic",
            gamma=8.0,              # inverse temperature: higher = more exploitative
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,   # epistemic curiosity
            learn_A=True,
            learn_B=False,
        )

        # Persistent state between episodes
        self._empirical_prior: list[jnp.ndarray] = list(D)
        self._last_qs: list[jnp.ndarray] | None = None
        self._last_action: jnp.ndarray | None = None
        self._episode_count: int = 0

    # ------------------------------------------------------------------

    def step(self, perf_obs: int, trend_obs: int) -> int:
        """
        Run one Active Inference cycle given new observations.
        Returns action index (LOWER/MAINTAIN/RAISE/ADD_BOT).
        """
        obs = [jnp.array([perf_obs]), jnp.array([trend_obs])]

        # 1. Perception: infer hidden states
        qs = self._agent.infer_states(obs, self._empirical_prior)

        # 2. Policy selection
        q_pi, _G = self._agent.infer_policies(qs)

        # 3. Sample action
        self._rng, rng_step = jax.random.split(self._rng)
        rng_batched = jax.random.split(rng_step, 1)   # (batch=1, 2)
        action = self._agent.sample_action(q_pi, rng_key=rng_batched)

        # 4. Update empirical prior for next step
        self._empirical_prior = self._agent.update_empirical_prior(action, qs)

        # 5. Online learning: update A (likelihood) from this observation
        beliefs_A = [jnp.expand_dims(qs[0], 1), jnp.expand_dims(qs[1], 1)]
        obs_seq   = [jnp.array([[perf_obs]]), jnp.array([[trend_obs]])]
        self._agent = self._agent.infer_parameters(beliefs_A, obs_seq, actions=None)

        self._last_qs = qs
        self._last_action = action
        self._episode_count += 1

        # action shape is (batch=1, n_factors=2); we only control factor 1 (difficulty)
        return int(action[0, 1])

    # ------------------------------------------------------------------

    def get_belief_summary(self) -> dict:
        """Return current beliefs as plain Python dicts for logging/display."""
        if self._last_qs is None:
            return {
                "engagement": {"FRUSTRATED": 1/3, "FLOW": 1/3, "BORED": 1/3},
                "difficulty_belief": [1/5] * 5,
                "episode": self._episode_count,
            }
        eng = np.array(self._last_qs[0][0, 0])   # (N_ENG,)
        diff = np.array(self._last_qs[1][0, 0])  # (N_DIFF,)
        return {
            "engagement": {
                "FRUSTRATED": float(eng[FRUSTRATED]),
                "FLOW":       float(eng[FLOW]),
                "BORED":      float(eng[BORED]),
            },
            "difficulty_belief": diff.tolist(),
            "episode": self._episode_count,
        }

    def dominant_state(self) -> str:
        summary = self.get_belief_summary()
        eng = summary["engagement"]
        return max(eng, key=eng.get)

    def reset(self, initial_skill: int = 3):
        """Start a fresh session (resets beliefs and prior)."""
        self._empirical_prior = list(_build_D(initial_skill))
        self._last_qs = None
        self._last_action = None
        self._episode_count = 0
