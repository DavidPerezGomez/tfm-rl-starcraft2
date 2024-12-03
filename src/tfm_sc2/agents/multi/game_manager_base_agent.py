from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

from pysc2.env.environment import TimeStep
from pysc2.lib import actions
from typing_extensions import override

from ..base_agent import BaseAgent

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS
from ...actions import AllActions, GameManagerActions
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
        self._proxy_agent = None
        self._prev_proxy_agent = None
        self._current_game_manager_action = None
        self._time_displacement = time_displacement
        self._completed_episode_steps = 0
        self._take_step = True

        # self._base_manager.exploit()
        # self._army_recruit_manager.exploit()
        # self._army_attack_manager.exploit()
        self._base_manager.setup_actions()
        self._army_recruit_manager.setup_actions()
        self._army_attack_manager.setup_actions()

    @override
    def train(self):
        super().train()
        self._base_manager.exploit()
        self._army_recruit_manager.exploit()
        self._army_attack_manager.exploit()

    @override
    def exploit(self):
        super().exploit()
        self._base_manager.exploit()
        self._army_recruit_manager.exploit()
        self._army_attack_manager.exploit()

    def fine_tune(self):
        super().train()
        self._base_manager.train()
        self._army_recruit_manager.train()
        self._army_attack_manager.train()

    @override
    def reset(self, **kwargs):
        super().reset()
        self._proxy_agent = None
        self._prev_proxy_agent = None
        self._current_game_manager_action = None
        self._completed_episode_steps = 0
        self._take_step = True

    @override
    def calculate_available_actions(self, obs: TimeStep) -> List[AllActions]:
        return self.agent_actions

    @override
    @abstractmethod
    def select_action(self, obs: TimeStep) -> GameManagerActions:
        pass

    def _select_proxy_agent(self):
        match self._current_game_manager_action:
            case GameManagerActions.EXPAND_BASE:
                self._proxy_agent = self._base_manager
            case GameManagerActions.EXPAND_ARMY:
                self._proxy_agent = self._army_recruit_manager
            case GameManagerActions.ATTACK:
                self._proxy_agent = self._army_attack_manager
            case _:
                raise RuntimeError(f"Unknown action: {self._current_game_manager_action.name}")

    @override
    def setup_positions(self, obs: TimeStep):
        super().setup_positions(obs)
        self._base_manager.setup_positions(obs)
        self._army_recruit_manager.setup_positions(obs)
        self._army_attack_manager.setup_positions(obs)

    @override
    def pre_step(self, obs: TimeStep, is_first_step: bool):
        self._take_step = obs.first() or self._completed_episode_steps % self._time_displacement == 0

        # self._base_manager.pre_step(obs, obs.first())
        # self._army_recruit_manager.pre_step(obs, obs.first())
        # self._army_attack_manager.pre_step(obs, obs.first())
        if self._take_step:
            super().pre_step(obs, obs.first())

    @override
    def update_command_center_positions(self):
        if self._take_step:
            super().update_command_center_positions()
        # self._base_manager.update_command_center_positions()
        # self._army_recruit_manager.update_command_center_positions()
        # self._army_attack_manager.update_command_center_positions()

    @override
    def update_supply_depot_positions(self):
        if self._take_step:
            super().update_supply_depot_positions()
        # self._base_manager.update_supply_depot_positions()
        # self._army_recruit_manager.update_supply_depot_positions()
        # self._army_attack_manager.update_supply_depot_positions()

    @override
    def update_barracks_positions(self):
        if self._take_step:
            super().update_barracks_positions()
        # self._base_manager.update_barracks_positions()
        # self._army_recruit_manager.update_barracks_positions()
        # self._army_attack_manager.update_barracks_positions()


    @override
    def step(self, obs: TimeStep, **kwargs):
        if obs.first():
            self.setup_positions(obs)

        self.pre_step(obs, obs.first())

        super().step(obs, only_super_step=True)

        if self._take_step:
            self.logger.debug("Taking game manager action")
            self._available_actions = self.calculate_available_actions(obs)
            self._current_game_manager_action = self.select_action(obs)
            self._select_proxy_agent()

        if self._prev_proxy_agent is not None:
            self._prev_proxy_agent.pre_step(obs, obs.first())

        if self._prev_proxy_agent != self._proxy_agent:
            self._proxy_agent.pre_step(obs, True)

        action, action_args, is_valid_action = self._proxy_agent.select_action(obs=obs)

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
            self._current_episode_stats.add_invalid_action(self._current_game_manager_action)
        else:
            self._current_episode_stats.add_valid_action(self._current_game_manager_action)
            self.logger.debug(f"[Step {self.steps}] Manager action: {self._current_game_manager_action.name} // Sub-agent action {action.name} (original action = {original_action})")

        self.post_step(obs, self._current_game_manager_action, None, self._current_game_manager_action, None, True)
        if not obs.last():
            self._proxy_agent.post_step(obs, action, action_args, original_action, original_action_args, is_valid_action)
        self._prev_proxy_agent = self._proxy_agent

        game_action = self._action_to_game[action]

        self._completed_episode_steps += 1

        if action_args is not None:
            return game_action(**action_args)

        return game_action()

    @override
    def save(self, checkpoint_path: Union[str|Path] = None):
        super().save(checkpoint_path)
        self._base_manager.save(self.checkpoint_path / "base_manager")
        self._army_recruit_manager.save(self.checkpoint_path / "army_recruit_manager")
        self._army_attack_manager.save(self.checkpoint_path / "army_attack_manager")
