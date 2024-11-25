from pysc2.env.environment import TimeStep
from typing_extensions import override
from typing import List

from ..dqn_agent import DQNAgent
from .with_army_attack_manager_actions import WithArmyAttackManagerActions
from ...actions import AllActions


class ArmyAttackManagerDQNAgent(WithArmyAttackManagerActions, DQNAgent):

    @override
    def available_actions(self, obs: TimeStep) -> List[AllActions]:
        available_actions = super().available_actions(obs)

        if len(available_actions) > 1 and AllActions.NO_OP in available_actions:
            available_actions = [a for a in available_actions if a != AllActions.NO_OP]

        return available_actions