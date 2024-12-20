import random
from typing import Any, Dict, List, Tuple

from pysc2.env.environment import TimeStep
from pysc2.lib import actions, units

from ...actions import (  # ResourceManagerActions,
    AllActions,
    ArmyAttackManagerActions,
    ArmyRecruitManagerActions,
    BaseManagerActions,
)
from ...constants import Constants
from ..base_agent import BaseAgent

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS
from ...types import AgentStage, Gas, Minerals


class SingleRandomAgent(BaseAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__original_agent_actions = list(
            set(
                list(BaseManagerActions) +
                list(ArmyRecruitManagerActions) +
                list(ArmyAttackManagerActions)
            ))
        self.__agent_actions = [a for a in self.__original_agent_actions if a in self._map_config["available_actions"]]

    @property
    def agent_actions(self) -> List[AllActions]:
        return self.__agent_actions

    def select_action(self, obs: TimeStep) -> Tuple[AllActions, Dict[str, Any]]:
        if self._action_masking:
            available_actions = self._available_actions
        else:
            available_actions = [a for a in self.agent_actions if a in self._map_config["available_actions"]]

        action = random.choice(available_actions)
        action_args, is_valid_action = self._get_action_args(obs=obs, action=action)

        return action, action_args, is_valid_action

    def _current_agent_stage(self):
        return AgentStage.EXPLOIT
