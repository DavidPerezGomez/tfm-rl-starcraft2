from abc import ABC, abstractmethod
from typing import List

from pysc2.env.environment import TimeStep
from pysc2.lib import actions, units
from typing_extensions import override

from ..base_agent import BaseAgent

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS
from ...actions import AllActions, ArmyRecruitManagerActions, GameManagerActions
from .with_army_attack_manager_actions import WithArmyAttackManagerActions
from .with_army_recruit_manager_actions import WithArmyRecruitManagerActions
from .with_base_manager_actions import WithBaseManagerActions
from .with_game_manager_actions import WithGameManagerActions


class GameManagerBaseAgent(WithGameManagerActions, BaseAgent, ABC):

    def __init__(self, base_manager: (BaseAgent, WithBaseManagerActions),
                 army_recruit_manager: (BaseAgent, WithArmyRecruitManagerActions),
                 army_attack_manager: (BaseAgent, WithArmyAttackManagerActions),
                 time_displacement: int = 5, **kwargs):
        super().__init__(**kwargs)
        self._base_manager = base_manager
        self._army_recruit_manager = army_recruit_manager
        self._army_attack_manager = army_attack_manager
        self._time_displacement = time_displacement

        self._base_manager.exploit()
        self._army_recruit_manager.exploit()
        self._army_attack_manager.exploit()
        self._base_manager.setup_actions()
        self._army_recruit_manager.setup_actions()
        self._army_attack_manager.setup_actions()

    def forward_action(self, obs: TimeStep, action: GameManagerActions):
        match action:
            case GameManagerActions.EXPAND_BASE:
                proxy_manager = self._base_manager
                actual_action, action_args, is_valid_action = proxy_manager.select_action(obs=obs)
            case GameManagerActions.EXPAND_ARMY:
                proxy_manager = self._army_recruit_manager
                actual_action, action_args, is_valid_action = proxy_manager.select_action(obs=obs)
            case GameManagerActions.ATTACK:
                proxy_manager = self._army_attack_manager
                actual_action, action_args, is_valid_action = proxy_manager.select_action(obs=obs)
            case _:
                raise RuntimeError(f"Unknown action: {action.name}")

        return actual_action, action_args, is_valid_action, proxy_manager

    @override
    def calculate_available_actions(self, obs: TimeStep) -> List[AllActions]:
        return self.agent_actions

    @override
    @abstractmethod
    def select_action(self, obs: TimeStep) -> GameManagerActions:
        pass

    def pre_step(self, obs: TimeStep):
        super().pre_step(obs)
        self._base_manager.pre_step(obs)
        self._army_recruit_manager.pre_step(obs)
        self._army_attack_manager.pre_step(obs)

    def step(self, obs: TimeStep, **kwargs):
        if obs.first():
            self.setup_positions()
            self._base_manager.setup_positions()
            self._army_recruit_manager.setup_positions()
            self._army_attack_manager.setup_positions()

        self.pre_step(obs)

        super().step(obs, only_super_step=True)

        self.update_supply_depot_positions()
        self._base_manager.update_supply_depot_positions()
        self._army_recruit_manager.update_supply_depot_positions()
        self._army_attack_manager.update_supply_depot_positions()

        self.update_command_center_positions()
        self._base_manager.update_supply_depot_positions()
        self._army_recruit_manager.update_supply_depot_positions()
        self._army_attack_manager.update_supply_depot_positions()

        self.update_barracks_positions()
        self._base_manager.update_barracks_positions()
        self._army_recruit_manager.update_barracks_positions()
        self._army_attack_manager.update_barracks_positions()

        self._available_actions = self.calculate_available_actions(obs)
        game_manager_action = self.select_action(obs)
        self.select_proxy_agent(game_manager_action)

        action, action_args, is_valid_action, proxy_manager = self.forward_action(obs=obs, action=game_manager_action)

        # if action == AllActions.NO_OP:
        #     self.logger.debug(f"Proxy manager for action {game_manager_action.name} returned a no-op, selecting a different action...")
        #     available_actions = [a for a in self._available_actions if a != game_manager_action]
        #     game_manager_action = self.select_action(obs, valid_actions=available_actions)
        #
        #     action, action_args, is_valid_action, proxy_manager = self.forward_action(obs=obs, action=game_manager_action)
        #     if action == AllActions.NO_OP:
        #         self.logger.debug(f"Proxy manager for action {game_manager_action.name} also returned a no-op, selecting the remaining action...")
        #         available_actions = [a for a in available_actions if a != game_manager_action]
        #         game_manager_action = available_actions[0]
        #         action, action_args, is_valid_action, proxy_manager = self.forward_action(obs=obs, action=game_manager_action)

        original_action = action
        original_action_args = action_args
        if not is_valid_action:
            self.logger.debug(f"Sub-agent action {action.name} is not valid anymore, returning NO_OP")
            action = AllActions.NO_OP
            action_args = None
            self._current_episode_stats.add_invalid_action(game_manager_action)
        else:
            self._current_episode_stats.add_valid_action(game_manager_action)
            self.logger.debug(f"[Step {self.steps}] Manager action: {game_manager_action.name} // Sub-agent action {action.name} (original action = {original_action})")

        self.post_step(obs, game_manager_action, None, game_manager_action, None, True)
        # self.post_step(obs, action, action_args, original_action, original_action_args, is_valid_action)

        game_action = self._action_to_game[action]

        if action_args is not None:
            return game_action(**action_args)

        return game_action()
