from pysc2.env.environment import TimeStep
from typing_extensions import override
from typing import List

from ..base_agent import BaseAgent
from .with_army_attack_manager_actions import WithArmyAttackManagerActions
from .with_random_action_selection import WithRandomActionSelection
from ...actions import AllActions


class ArmyAttackManagerRandomAgent(WithArmyAttackManagerActions, WithRandomActionSelection, BaseAgent):

    @override
    def calculate_available_actions(self, obs: TimeStep) -> List[AllActions]:
        available_actions = super().calculate_available_actions(obs)

        if len(available_actions) > 1 and AllActions.NO_OP in available_actions:
            available_actions = [a for a in available_actions if a != AllActions.NO_OP]

        return available_actions