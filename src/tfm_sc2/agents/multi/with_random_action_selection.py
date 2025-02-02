import random
from typing import Any, Dict, Tuple

from pysc2.env.environment import TimeStep
from pysc2.lib import actions

from ...actions import AllActions

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS

class WithRandomActionSelection:
    def select_action(self, obs: TimeStep, available_actions = None) -> Tuple[AllActions, Dict[str, Any]]:
        if self._action_masking:
            available_actions = self._available_actions
        else:
            available_actions = [a for a in self.agent_actions if a in self._map_config["available_actions"]]

        action = random.choice(available_actions)
        action_args, is_valid_action = self._get_action_args(obs=obs, action=action)

        return action, action_args, is_valid_action