"""
PlayerProxy implementations:
- BuiltInBotProxy: simple skill-scaled player controller
- ModelCheckpointProxy: pretrained Arnold model checkpoint wrapper
"""

import random
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import vizdoom as vzd

from src.interfaces import PlayerProxy as PlayerProxyInterface


class ArnoldDRQN(nn.Module):
    def __init__(self, input_channels: int = 4, num_actions: int = 29):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((6, 12))

        self.health_emb = nn.Embedding(11, 32)
        self.sel_ammo_emb = nn.Embedding(301, 32)

        self.proj_game_features = nn.Sequential(
            nn.Identity(),
            nn.Linear(4608, 512),
            nn.ReLU(inplace=True),
            nn.Identity(),
            nn.Linear(512, 2),
        )

        self.rnn = nn.LSTM(input_size=4672, hidden_size=512, batch_first=True)
        self.proj_action_scores = nn.Linear(512, num_actions)

    def forward(self, image, health_idx, ammo_idx, hx=None):
        x = self.conv(image)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        health_emb = self.health_emb(health_idx)
        ammo_emb = self.sel_ammo_emb(ammo_idx)

        rnn_input = torch.cat([x, health_emb, ammo_emb], dim=-1).unsqueeze(1)
        rnn_out, hx = self.rnn(rnn_input, hx)

        hidden = rnn_out[:, -1, :]
        action_logits = self.proj_action_scores(hidden)
        aux_out = self.proj_game_features(x)
        return action_logits, aux_out, hx


class ArnoldAgent:
    BUTTONS = [
        vzd.Button.ATTACK,
        vzd.Button.SPEED,
        vzd.Button.STRAFE,
        vzd.Button.MOVE_RIGHT,
        vzd.Button.MOVE_LEFT,
        vzd.Button.MOVE_BACKWARD,
        vzd.Button.MOVE_FORWARD,
        vzd.Button.TURN_RIGHT,
        vzd.Button.TURN_LEFT,
        vzd.Button.SELECT_WEAPON1,
        vzd.Button.SELECT_WEAPON2,
        vzd.Button.SELECT_WEAPON3,
        vzd.Button.SELECT_WEAPON4,
        vzd.Button.SELECT_WEAPON5,
        vzd.Button.SELECT_WEAPON6,
        vzd.Button.SELECT_NEXT_WEAPON,
        vzd.Button.SELECT_PREV_WEAPON,
        vzd.Button.LOOK_UP_DOWN_DELTA,
        vzd.Button.TURN_LEFT_RIGHT_DELTA,
        vzd.Button.MOVE_LEFT_RIGHT_DELTA,
    ]
    ACTION_SET: list[list[int]] = []

    @classmethod
    def ensure_action_set(cls) -> None:
        if cls.ACTION_SET:
            return

        def make_action(*buttons: int) -> list[int]:
            action = [0] * len(cls.BUTTONS)
            for button in buttons:
                if button in cls.BUTTONS:
                    action[cls.BUTTONS.index(button)] = 1
            return action

        # Simplified action space - basic movements and attack
        cls.ACTION_SET = [
            make_action(),  # NOOP
            make_action(vzd.Button.MOVE_LEFT),
            make_action(vzd.Button.MOVE_RIGHT),
            make_action(vzd.Button.MOVE_FORWARD),
            make_action(vzd.Button.MOVE_BACKWARD),
            make_action(vzd.Button.TURN_LEFT),
            make_action(vzd.Button.TURN_RIGHT),
            make_action(vzd.Button.ATTACK),
            make_action(vzd.Button.SPEED),
            make_action(vzd.Button.ATTACK, vzd.Button.MOVE_FORWARD),
            make_action(vzd.Button.ATTACK, vzd.Button.TURN_LEFT),
            make_action(vzd.Button.ATTACK, vzd.Button.TURN_RIGHT),
            make_action(vzd.Button.ATTACK, vzd.Button.MOVE_LEFT),
            make_action(vzd.Button.ATTACK, vzd.Button.MOVE_RIGHT),
        ]
        # Extend if needed for track_1
        while len(cls.ACTION_SET) < 35:
            cls.ACTION_SET.append(make_action())

    def __init__(self, skill_level: float, model_path: str, exploration_rate: float = 0.0):
        self.skill_level = float(np.clip(skill_level, 0.0, 1.0))
        self.exploration_rate = float(np.clip(exploration_rate, 0.0, 1.0))
        self.skill_random_rate = 0.35 * (1.0 - self.skill_level)
        self.random_action_prob = float(
            np.clip(self.exploration_rate + self.skill_random_rate, 0.0, 1.0))
        self.frame_skip = int(
            np.clip(round(1 + 3 * (1.0 - self.skill_level)), 1, 4))
        self._rng = np.random.RandomState(42)
        self._recurrent_state: Optional[tuple[torch.Tensor,
                                              torch.Tensor]] = None
        self._persistent_action: Optional[list[int]] = None
        self._persistence_counter = 0

        # Detect model architecture from path
        model_name = model_path.split('/')[-1]
        if 'track1' in model_name or 'track_1' in model_name:
            input_channels, num_actions = 3, 35
        else:
            input_channels, num_actions = 4, 29

        self.device = torch.device("cpu")
        self.model = ArnoldDRQN(input_channels=input_channels, num_actions=num_actions).to(self.device)
        self.model.eval()
        self._load_weights(model_path)

        if not self.ACTION_SET:
            self._build_action_set()

    def _load_weights(self, model_path: str) -> None:
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            result = self.model.load_state_dict(state_dict, strict=False)
            if result.missing_keys or result.unexpected_keys:
                print(
                    f"WARNING: checkpoint loaded with missing keys={result.missing_keys} "
                    f"unexpected keys={result.unexpected_keys}"
                )
        except Exception as exc:
            print(f"WARNING: Could not load checkpoint {model_path}: {exc}")
            self.model = None

    def _build_action_set(self) -> None:
        def make_action(*buttons: int) -> list[int]:
            action = [0] * len(self.BUTTONS)
            for button in buttons:
                if button in self.BUTTONS:
                    action[self.BUTTONS.index(button)] = 1
            return action

        # Simplified action space
        type(self).ACTION_SET = [
            make_action(),  # NOOP
            make_action(vzd.Button.MOVE_LEFT),
            make_action(vzd.Button.MOVE_RIGHT),
            make_action(vzd.Button.MOVE_FORWARD),
            make_action(vzd.Button.MOVE_BACKWARD),
            make_action(vzd.Button.TURN_LEFT),
            make_action(vzd.Button.TURN_RIGHT),
            make_action(vzd.Button.ATTACK),
            make_action(vzd.Button.SPEED),
            make_action(vzd.Button.ATTACK, vzd.Button.MOVE_FORWARD),
            make_action(vzd.Button.ATTACK, vzd.Button.TURN_LEFT),
            make_action(vzd.Button.ATTACK, vzd.Button.TURN_RIGHT),
            make_action(vzd.Button.ATTACK, vzd.Button.MOVE_LEFT),
            make_action(vzd.Button.ATTACK, vzd.Button.MOVE_RIGHT),
        ]
        # Extend if needed
        while len(type(self).ACTION_SET) < self.model.proj_action_scores.out_features:
            type(self).ACTION_SET.append(make_action())

    def reset(self) -> None:
        self._recurrent_state = None

    def action_for_state(self, game_state: Any) -> list[int]:
        if self.model is None or game_state is None:
            return self._random_action()

        if self._persistence_counter > 0 and self._persistent_action is not None:
            self._persistence_counter -= 1
            return self._persistent_action

        if self._rng.rand() < self.random_action_prob:
            action = self._random_action()
        else:
            action_idx = self._infer_action(game_state)
            action = self.ACTION_SET[action_idx]

        # Force more aggressive behavior - if action has ATTACK, also add MOVE_FORWARD
        if action[0] == 1:  # ATTACK button
            action[6] = 1  # MOVE_FORWARD

        self._persistent_action = action
        self._persistence_counter = self.frame_skip - 1
        return action

    def _random_action(self) -> list[int]:
        if not self.ACTION_SET:
            self._build_action_set()
        num_actions = self.model.proj_action_scores.out_features
        return self.ACTION_SET[self._rng.randint(num_actions)]

    def _infer_action(self, game_state: Any) -> int:
        image, health_idx, ammo_idx = self._prepare_game_state(game_state)
        with torch.no_grad():
            logits, _, self._recurrent_state = self.model(
                image.to(self.device),
                health_idx.to(self.device),
                ammo_idx.to(self.device),
                self._recurrent_state,
            )
        return int(logits.argmax(dim=-1).item())

    def _prepare_game_state(self, game_state: Any):
        screen = np.asarray(game_state.screen_buffer, dtype=np.float32) / 255.0
        if screen.ndim == 3:
            if screen.shape[0] in (3, 4):
                # already channel-first
                pass
            elif screen.shape[2] in (3, 4):
                screen = np.transpose(screen, (2, 0, 1))
            else:
                raise ValueError(
                    f"Unexpected screen buffer shape: {screen.shape}")
        elif screen.ndim == 2:
            screen = np.expand_dims(screen, 0)

        # Only pad to 4 channels if model expects it (deathmatch models)
        if screen.shape[0] == 3 and self.model.conv[0].in_channels == 4:
            pad = np.zeros(
                (1, screen.shape[1], screen.shape[2]), dtype=screen.dtype)
            screen = np.concatenate((screen, pad), axis=0)

        image = torch.from_numpy(screen).unsqueeze(0)

        game_vars = np.asarray(game_state.game_variables, dtype=np.float32)
        health = int(np.clip(game_vars[0] // 10, 0, 10))
        ammo = int(np.clip(game_vars[9], 0, 300))

        health_idx = torch.tensor([health], dtype=torch.long)
        ammo_idx = torch.tensor([ammo], dtype=torch.long)
        return image, health_idx, ammo_idx


class BuiltInBotProxy(PlayerProxyInterface):
    """Simple skill-scaled player proxy with heuristic action selection."""

    def __init__(
        self,
        skill_level: float = 0.5,
        exploration_rate: float = 0.1,
    ):
        self.skill_level = float(np.clip(skill_level, 0.0, 1.0))
        self.exploration_rate = float(np.clip(exploration_rate, 0.0, 1.0))
        self._rng = np.random.RandomState(42)
        self._id = f"builtin_bot_{int(self.skill_level * 100):03d}"

        ArnoldAgent.ensure_action_set()
        self._actions = ArnoldAgent.ACTION_SET

    @property
    def id(self) -> str:
        return self._id

    def act(self, obs: Any) -> list[int]:
        if self._rng.rand() < self.exploration_rate or obs is None:
            return self._random_action()

        health, ammo = self._parse_state(obs)
        low_health = health < 25
        if low_health:
            return self._recovery_action()

        if self.skill_level >= 0.75:
            if ammo > 0:
                return self._named_action([vzd.Button.ATTACK, vzd.Button.MOVE_FORWARD])
            return self._named_action([vzd.Button.MOVE_FORWARD])
        if self.skill_level >= 0.4:
            if ammo > 0:
                return self._named_action([vzd.Button.SPEED, vzd.Button.ATTACK])
            return self._named_action([vzd.Button.MOVE_FORWARD])
        if ammo > 0:
            return self._named_action([vzd.Button.ATTACK])
        return self._named_action([vzd.Button.MOVE_FORWARD])

    def _named_action(self, buttons: list[int]) -> list[int]:
        for action in self._actions:
            if all(action[ArnoldAgent.BUTTONS.index(button)] == 1 for button in buttons):
                return action
        return self._random_action()

    def _random_action(self) -> list[int]:
        return self._actions[self._rng.randint(len(self._actions))]

    def _recovery_action(self) -> list[int]:
        return self._named_action([vzd.Button.SPEED, vzd.Button.MOVE_BACKWARD])

    def _parse_state(self, obs: Any) -> tuple[float, float]:
        if hasattr(obs, "game_variables"):
            vars = obs.game_variables
        elif isinstance(obs, dict):
            vars = obs.get("game_variables", [])
        else:
            vars = []

        health = float(vars[0]) if len(vars) > 0 else 100.0
        ammo = float(vars[9]) if len(vars) > 9 else 0.0
        return health, ammo


class ModelCheckpointProxy(PlayerProxyInterface):
    """Player proxy that uses a pretrained Arnold checkpoint."""

    def __init__(
        self,
        model_path: str,
        skill_level: float = 0.5,
        exploration_rate: float = 0.05,
    ):
        self._id = f"model_proxy_{model_path.split('/')[-1]}"
        self.skill_level = float(np.clip(skill_level, 0.0, 1.0))
        self.exploration_rate = float(np.clip(exploration_rate, 0.0, 1.0))
        self._agent = ArnoldAgent(
            self.skill_level, model_path, exploration_rate=self.exploration_rate)

    @property
    def id(self) -> str:
        return self._id

    def act(self, obs: Any) -> list[int]:
        if obs is None:
            return self._agent._random_action()
        return self._agent.action_for_state(obs)
