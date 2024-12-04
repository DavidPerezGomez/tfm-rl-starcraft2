from typing import Any, Dict, Tuple

from pysc2.env.environment import TimeStep
from pysc2.lib import actions
from typing_extensions import override

from ...actions import AllActions
from ..dqn_agent import DQNAgent

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS
from .game_manager_base_agent import GameManagerBaseAgent


class GameManagerDQNAgent(GameManagerBaseAgent, DQNAgent):

    @override
    def burnin(self):
        super().burnin()
        if hasattr(self._base_manager, "burnin") and callable(getattr(self._base_manager, "burning")):
            self._base_manager.burnin()
        if hasattr(self._army_recruit_manager, "burnin") and callable(getattr(self._army_recruit_manager, "burning")):
            self._army_recruit_manager.burnin()
        if hasattr(self._army_attack_manager, "burnin") and callable(getattr(self._army_attack_manager, "burning")):
            self._army_attack_manager.burnin()

    @override
    @property
    def memory_replay_ready(self) -> bool:
        return super().memory_replay_ready \
                and (not hasattr(self._base_manager, "memory_replay_ready") or self._base_manager.memory_replay_ready) \
                and (not hasattr(self._army_attack_manager, "memory_replay_ready") or self._army_attack_manager.memory_replay_ready) \
                and (not hasattr(self._army_recruit_manager, "memory_replay_ready") or self._army_recruit_manager.memory_replay_ready) \

    def select_action(self, obs: TimeStep) -> Tuple[AllActions, Dict[str, Any]]:
        valid_actions = self._available_actions
        if valid_actions is not None:
            valid_actions = self._actions_to_network(valid_actions)
        if (self._random_mode) or (self._train and self._burnin):
            if not self._status_flags["burnin_started"]:
                self.logger.info(f"Starting burn-in")
                self._status_flags["burnin_started"] = True
            if self._random_mode:
                self.logger.debug(f"Random mode - collecting experience from random actions")
            else:
                self.logger.debug(f"Burn in capacity: {100 * self._buffer.burn_in_capacity:.2f}%")
            raw_action = self.main_network.get_random_action(valid_actions=valid_actions)
            # raw_action = self.main_network.get_random_action()
        elif self.is_training:
            if not self._status_flags["train_started"]:
                self.logger.info(f"Starting training")
                self._status_flags["train_started"] = True
            raw_action = self.main_network.get_action(self._current_state_tuple, epsilon=self.epsilon, valid_actions=valid_actions)
        else:
            if not self._status_flags["exploit_started"]:
                self.logger.info(f"Starting exploit")
                self._status_flags["exploit_started"] = True
            raw_action = self.main_network.get_greedy_action(self._current_state_tuple, valid_actions=valid_actions)

        # Convert the "raw" action to a the right type of action
        action = self._idx_to_action[raw_action]

        return action
