"""
Game variable definitions and groupings for VizDoom deathmatch recording.

Groups can be composed freely:
    my_vars = list(COMBAT) + list(NAVIGATION)

Each group exposes `.column_names` — use these to slice a recorded DataFrame:
    df[COMBAT.column_names]
"""

from dataclasses import dataclass
from typing import List, Tuple
import vizdoom as vzd


@dataclass(frozen=True)
class VariableGroup:
    name: str
    variables: Tuple[vzd.GameVariable, ...]

    def __iter__(self):
        return iter(self.variables)

    def __len__(self):
        return len(self.variables)

    def __add__(self, other: "VariableGroup") -> "VariableGroup":
        return VariableGroup(
            name=f"{self.name}+{other.name}",
            variables=self.variables + other.variables,
        )

    @property
    def column_names(self) -> List[str]:
        return [v.name.lower() for v in self.variables]


# ── Groups ────────────────────────────────────────────────────────────────────

COMBAT = VariableGroup("combat", (
    vzd.GameVariable.KILLCOUNT,        # monsters/players killed this episode
    vzd.GameVariable.DEATHCOUNT,       # times player died
    vzd.GameVariable.FRAGCOUNT,        # net frag score (kills - deaths)
    vzd.GameVariable.HITCOUNT,         # successful hits dealt
    vzd.GameVariable.HITS_TAKEN,       # hits received
    vzd.GameVariable.DAMAGECOUNT,      # total damage dealt
    vzd.GameVariable.DAMAGE_TAKEN,     # total damage received
    vzd.GameVariable.ATTACK_READY,     # primary fire ready (0/1)
    vzd.GameVariable.ALTATTACK_READY,  # alt fire ready (0/1)
))

NAVIGATION = VariableGroup("navigation", (
    vzd.GameVariable.POSITION_X,    # world X coordinate
    vzd.GameVariable.POSITION_Y,    # world Y coordinate
    vzd.GameVariable.POSITION_Z,    # world Z coordinate
    vzd.GameVariable.ANGLE,         # yaw angle (degrees)
    vzd.GameVariable.PITCH,         # pitch angle (degrees)
    vzd.GameVariable.VELOCITY_X,    # X velocity
    vzd.GameVariable.VELOCITY_Y,    # Y velocity
    vzd.GameVariable.VELOCITY_Z,    # Z velocity (jump/fall)
    vzd.GameVariable.ON_GROUND,     # is player on ground (0/1)
))

PLAYER_STATE = VariableGroup("player_state", (
    vzd.GameVariable.HEALTH,       # current health (100 = full)
    vzd.GameVariable.ARMOR,        # current armor
    vzd.GameVariable.DEAD,         # is player dead (0/1)
    vzd.GameVariable.ITEMCOUNT,    # items collected this episode
    vzd.GameVariable.VIEW_HEIGHT,  # camera height (crouch indicator)
))

# Selected weapon + full per-slot inventory (weapons and ammo for each slot)
WEAPONS_BASIC = VariableGroup("weapons_basic", (
    vzd.GameVariable.SELECTED_WEAPON,      # currently held weapon slot (0–9)
    vzd.GameVariable.SELECTED_WEAPON_AMMO, # ammo for current weapon
))

WEAPONS_FULL = VariableGroup("weapons_full", (
    vzd.GameVariable.SELECTED_WEAPON,
    vzd.GameVariable.SELECTED_WEAPON_AMMO,
    # Weapon ownership per slot (0 = not owned, 1 = owned)
    vzd.GameVariable.WEAPON0,
    vzd.GameVariable.WEAPON1,
    vzd.GameVariable.WEAPON2,
    vzd.GameVariable.WEAPON3,
    vzd.GameVariable.WEAPON4,
    vzd.GameVariable.WEAPON5,
    vzd.GameVariable.WEAPON6,
    vzd.GameVariable.WEAPON7,
    vzd.GameVariable.WEAPON8,
    vzd.GameVariable.WEAPON9,
    # Ammo count per type (types 0–9 map to: fist/pistol, bullets, shells, rockets,
    # cells, cells, unused, unused, unused, unused)
    vzd.GameVariable.AMMO0,
    vzd.GameVariable.AMMO1,
    vzd.GameVariable.AMMO2,
    vzd.GameVariable.AMMO3,
    vzd.GameVariable.AMMO4,
    vzd.GameVariable.AMMO5,
    vzd.GameVariable.AMMO6,
    vzd.GameVariable.AMMO7,
    vzd.GameVariable.AMMO8,
    vzd.GameVariable.AMMO9,
))

# Per-player frag counts (up to 4 players; expand to PLAYER8 for larger matches)
MULTIPLAYER = VariableGroup("multiplayer", (
    vzd.GameVariable.PLAYER_COUNT,       # total players in game
    vzd.GameVariable.PLAYER_NUMBER,      # this player's slot number
    vzd.GameVariable.PLAYER1_FRAGCOUNT,
    vzd.GameVariable.PLAYER2_FRAGCOUNT,
    vzd.GameVariable.PLAYER3_FRAGCOUNT,
    vzd.GameVariable.PLAYER4_FRAGCOUNT,
))

# ── Compound sets ─────────────────────────────────────────────────────────────

ALL_GROUPS = (COMBAT, NAVIGATION, PLAYER_STATE, WEAPONS_FULL, MULTIPLAYER)

# Default comprehensive variable list — no duplicates across the groups above
DEATHMATCH_VARS: List[vzd.GameVariable] = [
    var for group in ALL_GROUPS for var in group
]
