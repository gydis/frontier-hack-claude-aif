"""
Dashboard — a separate pygame window that visualises DM state alongside VizDoom.

Shows:
  - Current dominant player state with color coding (red/green/yellow)
  - Belief bar chart over engagement states
  - Skill level history sparkline
  - Last DM action

Run in the same process; call update() after each episode decision.
"""

import pygame

WIDTH, HEIGHT = 400, 320
FPS = 10

# Colors
BG        = (20, 20, 30)
TEXT      = (220, 220, 220)
DIM_TEXT  = (120, 120, 140)
FRUS_COL  = (220, 60, 60)
FLOW_COL  = (60, 200, 100)
BORED_COL = (220, 180, 40)
BAR_COL   = (80, 140, 220)
GRID_COL  = (40, 40, 55)
SKILL_COL = (180, 100, 220)


class Dashboard:
    """
    Optional pygame overlay.  Create once, call update() per episode.
    Call close() at session end.
    """

    def __init__(self, title: str = "Dungeon Master — Active Inference DDA"):
        pygame.init()
        self._screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(title)
        self._clock = pygame.time.Clock()
        self._font_large = pygame.font.SysFont("monospace", 22, bold=True)
        self._font_med   = pygame.font.SysFont("monospace", 15)
        self._font_small = pygame.font.SysFont("monospace", 12)

        self._state: str = "?"
        self._beliefs: dict = {"FRUSTRATED": 1/3, "FLOW": 1/3, "BORED": 1/3}
        self._action_desc: str = "—"
        self._skill_history: list[int] = []
        self._perf_history: list[int] = []    # 0=LOW, 1=MED, 2=HIGH

    def update(self, dominant_state: str, beliefs: dict, action_desc: str,
               skill: int, perf_obs: int):
        """Call once per episode with fresh DM data."""
        self._state = dominant_state
        self._beliefs = beliefs.get("engagement", beliefs)
        self._action_desc = action_desc
        self._skill_history.append(skill)
        self._perf_history.append(perf_obs)
        if len(self._skill_history) > 30:
            self._skill_history = self._skill_history[-30:]
            self._perf_history  = self._perf_history[-30:]
        self._draw()

    def _draw(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        self._screen.fill(BG)

        y = 10
        # Title
        self._blit(f"  Active Inference Dungeon Master", self._font_med, DIM_TEXT, (10, y))
        y += 22

        # State badge
        col = {"FRUSTRATED": FRUS_COL, "FLOW": FLOW_COL, "BORED": BORED_COL}.get(self._state, TEXT)
        pygame.draw.rect(self._screen, col, (10, y, WIDTH - 20, 36), border_radius=6)
        label = f"Player state: {self._state}"
        surf = self._font_large.render(label, True, (10, 10, 10))
        self._screen.blit(surf, (20, y + 7))
        y += 46

        # Belief bars
        self._blit("Engagement beliefs:", self._font_med, DIM_TEXT, (10, y))
        y += 18
        for name, val in [("FRUSTRATED", self._beliefs.get("FRUSTRATED", 0)),
                          ("FLOW",       self._beliefs.get("FLOW", 0)),
                          ("BORED",      self._beliefs.get("BORED", 0))]:
            bar_w = int(val * (WIDTH - 120))
            bar_col = {"FRUSTRATED": FRUS_COL, "FLOW": FLOW_COL, "BORED": BORED_COL}.get(name, BAR_COL)
            pygame.draw.rect(self._screen, GRID_COL, (90, y, WIDTH - 100, 16), border_radius=3)
            if bar_w > 0:
                pygame.draw.rect(self._screen, bar_col, (90, y, bar_w, 16), border_radius=3)
            self._blit(f"{name[:4]}  {val:.2f}", self._font_small, TEXT, (10, y + 1))
            y += 20
        y += 6

        # Action
        self._blit(f"Last action: {self._action_desc}", self._font_med, TEXT, (10, y))
        y += 22

        # Skill sparkline
        self._blit("Skill level history:", self._font_med, DIM_TEXT, (10, y))
        y += 18
        self._draw_sparkline(self._skill_history, (10, y, WIDTH - 20, 50), 1, 5, SKILL_COL)
        y += 56

        # Episode count
        ep = len(self._skill_history)
        if self._skill_history:
            self._blit(f"Episode {ep}  |  Current skill: {self._skill_history[-1]}", self._font_small, DIM_TEXT, (10, y))

        pygame.display.flip()
        self._clock.tick(FPS)

    def _draw_sparkline(self, values: list, rect: tuple, vmin: float, vmax: float, color):
        x0, y0, w, h = rect
        pygame.draw.rect(self._screen, GRID_COL, rect, border_radius=3)
        if len(values) < 2:
            return
        n = len(values)
        pts = []
        for i, v in enumerate(values):
            px = x0 + int(i / (n - 1) * w)
            py = y0 + h - int((v - vmin) / max(vmax - vmin, 1) * h)
            py = max(y0, min(y0 + h, py))
            pts.append((px, py))
        if len(pts) >= 2:
            pygame.draw.lines(self._screen, color, False, pts, 2)

    def _blit(self, text: str, font, color, pos):
        surf = font.render(text, True, color)
        self._screen.blit(surf, pos)

    def close(self):
        pygame.quit()
