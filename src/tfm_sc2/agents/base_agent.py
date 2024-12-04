import logging
import pickle
import random
import time
from abc import ABC, abstractmethod
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch_directml
from codecarbon.emissions_tracker import BaseEmissionsTracker
from pysc2.agents import base_agent
from pysc2.env.environment import TimeStep
from pysc2.lib import actions, features, units
from pysc2.lib.features import PlayerRelative
from typing_extensions import Self, deprecated

from ..actions import AllActions
from ..constants import Constants, SC2Costs
from ..types import AgentStage, Minerals, Position, RewardMode, State
from ..with_logger import WithLogger
from .stats import AgentStats, AggregatedEpisodeStats, EpisodeStats


class BaseAgent(WithLogger, ABC, base_agent.BaseAgent):
    _AGENT_FILE: str = "agent.pkl"
    _STATS_FILE: str = "stats.parquet"

    _action_to_game = {
        AllActions.NO_OP: actions.RAW_FUNCTIONS.no_op,
        AllActions.HARVEST_MINERALS: lambda source_unit_tag, target_unit_tag: actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", source_unit_tag, target_unit_tag),
        AllActions.RECRUIT_SCV_0: lambda source_unit_tag: actions.RAW_FUNCTIONS.Train_SCV_quick("now", source_unit_tag),
        AllActions.RECRUIT_SCV_1: lambda source_unit_tag: actions.RAW_FUNCTIONS.Train_SCV_quick("now", source_unit_tag),
        AllActions.RECRUIT_SCV_2: lambda source_unit_tag: actions.RAW_FUNCTIONS.Train_SCV_quick("now", source_unit_tag),
        AllActions.BUILD_SUPPLY_DEPOT: lambda source_unit_tag, target_position: actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", source_unit_tag, target_position),
        AllActions.BUILD_COMMAND_CENTER: lambda source_unit_tag, target_position: actions.RAW_FUNCTIONS.Build_CommandCenter_pt("now", source_unit_tag, target_position),
        AllActions.BUILD_BARRACKS: lambda source_unit_tag, target_position: actions.RAW_FUNCTIONS.Build_Barracks_pt("now", source_unit_tag, target_position),
        AllActions.RECRUIT_MARINE: lambda source_unit_tag: actions.RAW_FUNCTIONS.Train_Marine_quick("now", source_unit_tag),
        AllActions.ATTACK_CLOSEST_BUILDING: lambda source_unit_tag, target_unit_tag: actions.RAW_FUNCTIONS.Attack_unit("now", source_unit_tag, target_unit_tag),
        AllActions.ATTACK_CLOSEST_WORKER: lambda source_unit_tag, target_unit_tag: actions.RAW_FUNCTIONS.Attack_unit("now", source_unit_tag, target_unit_tag),
        AllActions.ATTACK_CLOSEST_ARMY: lambda source_unit_tag, target_unit_tag: actions.RAW_FUNCTIONS.Attack_unit("now", source_unit_tag, target_unit_tag),
        AllActions.ATTACK_BUILDINGS: lambda source_unit_tag, target_unit_tag: actions.RAW_FUNCTIONS.Attack_unit("now", source_unit_tag, target_unit_tag),
        AllActions.ATTACK_WORKERS: lambda source_unit_tag, target_unit_tag: actions.RAW_FUNCTIONS.Attack_unit("now", source_unit_tag, target_unit_tag),
        AllActions.ATTACK_ARMY: lambda source_unit_tag, target_unit_tag: actions.RAW_FUNCTIONS.Attack_unit("now", source_unit_tag, target_unit_tag),
    }

    def __init__(self,
                 map_name: str, map_config: Dict,
                 train: bool = True, action_masking: bool = False, tracker_update_freq_seconds: int = 10,
                 reward_mode: RewardMode = RewardMode.REWARD, score_method: str = "get_reward_as_score",
                 **kwargs):
        super().__init__(**kwargs)
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch_directml.device():
            self.device = torch_directml.device()
        else:
            self.device = 'cpu'

        self._map_name = map_name
        self._map_config = map_config
        self._supply_depot_positions = None
        self._command_center_positions = None
        self._command_center_0_pos = None
        self._command_center_1_pos = None
        self._command_center_2_pos = None
        self._barrack_positions = None
        self._used_supply_depot_positions = None
        self._used_command_center_positions = None
        self._used_barrack_positions = None
        self._train = train
        self._exploit = not train
        self._action_masking = action_masking
        self._tracker: BaseEmissionsTracker = None
        self._tracker_update_freq_seconds = tracker_update_freq_seconds
        self._tracker_last_update = time.time()
        self._prev_action = None
        self._prev_action_args = None
        self._prev_action_is_valid = None
        self._prev_score = 0.
        self._current_score = 0.
        self._current_reward = 0.
        self._current_adjusted_reward = 0.
        self._reward_mode = reward_mode
        self._score_method = score_method
        self._available_actions = None
        self._current_state_tuple = None
        self._prev_state_tuple = None
        self._current_obs_unit_info = None
        self._prev_minerals = 0
        self._prev_army_spending = 0
        self._prev_diff_marines = 0
        self._prev_health_difference_score = 0
        self._prev_game_score = 0

        self._action_to_idx = None
        self._idx_to_action = None
        self._num_actions = None
        self._best_mean_rewards = None

        self._status_flags = dict(
            train_started=False,
            exploit_started=False,
        )

        self.initialize()

    @property
    def _collect_stats(self) -> bool:
        return True

    def setup_actions(self):
        self._idx_to_action = { idx: action for idx, action in enumerate(self.agent_actions) }
        self._action_to_idx = { action: idx for idx, action in enumerate(self.agent_actions) }
        self._num_actions = len(self.agent_actions)

    def initialize(self):
        self._current_episode_stats = EpisodeStats(map_name=self._map_name)
        self._episode_stats = {self._map_name: []}
        self._aggregated_episode_stats = {self._map_name: AggregatedEpisodeStats(map_name=self._map_name)}
        self._agent_stats = {self._map_name: AgentStats(map_name=self._map_name)}

    def set_tracker(self, tracker: BaseEmissionsTracker):
        self._tracker = tracker
        self._tracker_last_update = time.time()

    @property
    def current_agent_stats(self) -> AgentStats:
        return self._agent_stats[self._map_name]

    @property
    def current_aggregated_episode_stats(self) -> AggregatedEpisodeStats:
        return self._aggregated_episode_stats[self._map_name]

    def save(self, checkpoint_path: Union[str|Path] = None):
        if checkpoint_path is None:
            raise RuntimeError(f"No checkpoint path was provided to save")

        if not checkpoint_path.exists():
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        agent_attrs = self._get_agent_attrs()
        agent_path = checkpoint_path / self._AGENT_FILE

        with open(agent_path, "wb") as f:
            pickle.dump(agent_attrs, f)
            self.logger.info(f"Saved agent attributes to {agent_path}")

        self.save_stats(checkpoint_path)

    @classmethod
    def load(cls, checkpoint_path: Union[str|Path], map_name: str, map_config: Dict, **kwargs) -> Self:
        checkpoint_path = Path(checkpoint_path)
        agent_attrs_file = checkpoint_path / cls._AGENT_FILE
        with open(agent_attrs_file, mode="rb") as f:
            agent_attrs = pickle.load(f)

        init_attrs = cls._extract_init_arguments(agent_attrs=agent_attrs, map_name=map_name, map_config=map_config)
        agent = cls(**init_attrs, **kwargs)
        agent._load_agent_attrs(agent_attrs)

        if agent._current_episode_stats is None or agent._current_episode_stats.map_name != map_name:
            agent._current_episode_stats = EpisodeStats(map_name=map_name)
        if map_name not in agent._agent_stats:
            agent._agent_stats[map_name] = AgentStats(map_name=map_name)
        if map_name not in agent._aggregated_episode_stats:
            agent._aggregated_episode_stats[map_name] = AggregatedEpisodeStats(map_name=map_name)
        if map_name not in agent._episode_stats:
            agent._episode_stats[map_name] = []

        return agent

    def save_stats(self, checkpoint_path: Union[str|Path] = None):
        if checkpoint_path is None:
            raise RuntimeError(f"No checkpoint path was provided to save")

        def _add_dummy_action(stats, count_cols):
            for count_col in count_cols:
                stats[count_col] = stats[count_col].apply(lambda d: dict(dummy=0) if len(d) == 0 else d)
            return stats

        try:
            all_episode_stats = [v for v in self._episode_stats.values()]
            all_episode_stats = reduce(lambda v1, v2: v1 + v2, all_episode_stats)
            episode_stats_pd = pd.DataFrame(data=all_episode_stats)
            if any(episode_stats_pd):
                episode_stats_pd = _add_dummy_action(episode_stats_pd, ["invalid_action_counts", "valid_action_counts"])
            episode_stats_pd.to_parquet(checkpoint_path / "episode_stats.parquet")
            self.logger.info(f"Saved episode stats to {checkpoint_path}")
        except Exception as error:
            self.logger.error(f"Error saving episode stats")
            self.logger.exception(error)

        try:
            all_agent_stats = [v for v in self._agent_stats.values()]
            agent_stats_pd = pd.DataFrame(data=all_agent_stats)
            agent_stats_pd = _add_dummy_action(
                agent_stats_pd,
                ["invalid_action_counts", "valid_action_counts", "invalid_action_counts_per_stage", "valid_action_counts_per_stage"])
            agent_stats_pd.to_parquet(checkpoint_path / "agent_stats.parquet")
            self.logger.info(f"Saved agent stats to {checkpoint_path}")
        except Exception as error:
            self.logger.error(f"Error saving agent stats")
            self.logger.exception(error)

        try:
            all_aggregated_stats = [v for v in self._aggregated_episode_stats.values()]
            aggregated_stats_pd = pd.DataFrame(data=all_aggregated_stats)
            aggregated_stats_pd = _add_dummy_action(
                aggregated_stats_pd,
                ["invalid_action_counts", "valid_action_counts", "invalid_action_counts_per_stage", "valid_action_counts_per_stage"])
            aggregated_stats_pd.to_parquet(checkpoint_path / "aggregated_stats.parquet")
            self.logger.info(f"Saved aggregated stats to {checkpoint_path}")
        except Exception as error:
            self.logger.error(f"Error saving aggregated stats")
            self.logger.exception(error)

    @classmethod
    def _extract_init_arguments(cls, agent_attrs: Dict[str, Any], map_name: str, map_config: Dict) -> Dict[str, Any]:
        return dict(
            map_name=map_name,
            map_config=map_config,
            train=agent_attrs["train"],
            log_name=agent_attrs["log_name"],
        )

    def _load_agent_attrs(self, agent_attrs: Dict):
        self._train = agent_attrs["train"]
        self._exploit = agent_attrs.get("exploit", not self._train)
        self._action_masking = agent_attrs["action_masking"]
        self._agent_stats = agent_attrs["agent_stats"]
        self._episode_stats = agent_attrs["episode_stats"]
        self._aggregated_episode_stats = agent_attrs["aggregated_episode_stats"]
        # From SC2's Base agent
        self.reward = agent_attrs["reward"]
        self.episodes = agent_attrs["episodes"]
        self.steps = agent_attrs["steps"]
        self.obs_spec = agent_attrs["obs_spec"]
        self.action_spec = agent_attrs["action_spec"]
        self._reward_mode = agent_attrs["reward_mode"]
        self._score_method = agent_attrs["score_method"]
        # From WithLogger
        self._log_name = agent_attrs["log_name"]
        if self.logger.name != self._log_name:
            self._logger = logging.getLogger(self._log_name)

    def _get_agent_attrs(self):
        return dict(
            train=self._train,
            exploit=self._exploit,
            action_masking=self._action_masking,
            agent_stats=self._agent_stats,
            episode_stats=self._episode_stats,
            aggregated_episode_stats=self._aggregated_episode_stats,
            # From SC2's Base agent
            reward=self.reward,
            episodes=self.episodes,
            steps=self.steps,
            obs_spec=self.obs_spec,
            action_spec=self.action_spec,
            reward_mode=self._reward_mode,
            score_method=self._score_method,
            # From logger
            log_name=self._log_name,
        )

    def train(self):
        """Set the agent in training mode."""
        self._train = True
        self._exploit = False

    def exploit(self):
        """Set the agent in training mode."""
        self._train = False
        self._exploit = True

    @property
    def action_masking(self) -> bool:
        return self._action_masking

    def set_action_masking(self, action_masking: bool):
        self._action_masking = action_masking

    def _current_agent_stage(self):
        if self._exploit:
            return AgentStage.EXPLOIT
        if self.is_training:
            return AgentStage.TRAINING
        return AgentStage.UNKNOWN

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._supply_depot_positions = None
        self._barrack_positions = None
        self._command_center_positions = None
        self._used_supply_depot_positions = None
        self._used_command_center_positions = None
        self._used_barrack_positions = None
        self._prev_action = None
        self._prev_action_args = None
        self._prev_action_is_valid = None
        self._prev_score = 0.
        self._current_score = 0.
        self._current_reward = 0.
        self._current_adjusted_reward = 0.
        self._prev_state_tuple = None
        self._available_actions = None
        self._current_state_tuple = None
        self._current_obs_unit_info = None
        self._prev_minerals = 0
        self._prev_army_spending = 0
        self._prev_diff_marines = 0
        self._prev_health_difference_score = 0
        self._prev_game_score = 0

        current_stage = self._current_agent_stage().name
        self._current_episode_stats = EpisodeStats(map_name=self._map_name, is_burnin=False, is_training=self.is_training, is_exploit=self._exploit, episode=self.current_agent_stats.episode_count, initial_stage=current_stage)
        self.current_agent_stats.episode_count += 1
        self.current_agent_stats.episode_count_per_stage[current_stage] += 1

    @property
    def is_training(self):
        return self._train

    def setup_positions(self, obs: TimeStep):
        if self._map_name == "Simple64":
            ally_unit = next(u for u in obs.observation.raw_units if u.alliance==PlayerRelative.SELF)
            position = "top_left" if ally_unit.y < 50 else "bottom_right"
            self.logger.debug(f"Map {self._map_name} - Started at '{position}' position")
            self._supply_depot_positions = self._map_config["positions"][position].get(units.Terran.SupplyDepot, []).copy()
            self._command_center_positions = self._map_config["positions"][position].get(units.Terran.CommandCenter, []).copy()
            self._barrack_positions = self._map_config["positions"][position].get(units.Terran.Barracks, []).copy()
        else:
            self._supply_depot_positions = self._map_config["positions"].get(units.Terran.SupplyDepot, []).copy()
            self._command_center_positions = self._map_config["positions"].get(units.Terran.CommandCenter, []).copy()
            self._barrack_positions = self._map_config["positions"].get(units.Terran.Barracks, []).copy()

        self._supply_depot_positions = [Position(t[0], t[1]) for t in self._supply_depot_positions]
        self._command_center_positions = [Position(t[0], t[1]) for t in self._command_center_positions]
        self._command_center_0_pos = self._command_center_positions[0] if len(self._command_center_positions) > 0 else None
        self._command_center_1_pos = self._command_center_positions[1] if len(self._command_center_positions) > 1 else None
        self._command_center_2_pos = self._command_center_positions[2] if len(self._command_center_positions) > 2 else None
        self._barrack_positions = [Position(t[0], t[1]) for t in self._barrack_positions]

    @property
    @abstractmethod
    def agent_actions(self) -> List[AllActions]:
        pass

    @abstractmethod
    def select_action(self, obs: TimeStep) -> Tuple[AllActions, Dict[str, Any]]:
        pass
        # return actions.FUNCTIONS.no_op()

    def _get_action_args(self, obs: TimeStep, action: AllActions) -> Dict[str, any]:
        try:
            match action:
                case AllActions.NO_OP:
                    action_args = None
                case AllActions.HARVEST_MINERALS:
                    minerals = [unit for unit in self._get_units(alliances=PlayerRelative.NEUTRAL) if Minerals.contains(unit.unit_type)]
                    assert (len(minerals) > 0), "There are no minerals to harvest"
                    command_centers = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.CommandCenter)
                    assert (len(command_centers) > 0), "There are no command centers"
                    idle_workers = self.get_idle_workers()
                    assert (len(idle_workers) > 0), "There are no idle workers"

                    worker = random.choice(idle_workers)
                    command_center, _ = self._get_closest(command_centers, Position(worker.x, worker.y))
                    mineral, _ = self._get_closest(minerals, Position(command_center.x, command_center.y))
                    action_args = dict(source_unit_tag=worker.tag, target_unit_tag=mineral.tag)
                case AllActions.RECRUIT_SCV_0:
                    assert self._command_center_0_pos is not None, "There is no position for first command center in the map"
                    command_center = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.CommandCenter, positions=[self._command_center_0_pos], completed_only=True, first_only=True)
                    assert command_center is not None, "There is no first command center"
                    assert command_center.order_length < Constants.COMMAND_CENTER_QUEUE_LENGTH, "Command center's queue is full"
                    action_args = dict(source_unit_tag=command_center.tag)
                case AllActions.RECRUIT_SCV_1:
                    assert self._command_center_1_pos is not None, "There is no position for second command center in the map"
                    command_center = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.CommandCenter, positions=[self._command_center_1_pos], completed_only=True, first_only=True)
                    assert command_center is not None, "There is no second command center"
                    assert command_center.order_length < Constants.COMMAND_CENTER_QUEUE_LENGTH, "Command center's queue is full"
                    action_args = dict(source_unit_tag=command_center.tag)
                case AllActions.RECRUIT_SCV_2:
                    assert self._command_center_2_pos is not None, "There is no position for third command center in the map"
                    command_center = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.CommandCenter, positions=[self._command_center_2_pos], completed_only=True, first_only=True)
                    assert command_center is not None, "There is no third command center"
                    assert command_center.order_length < Constants.COMMAND_CENTER_QUEUE_LENGTH, "Command center's queue is full"
                    action_args = dict(source_unit_tag=command_center.tag)
                case AllActions.BUILD_SUPPLY_DEPOT:
                    position = self.get_next_supply_depot_position()
                    workers = self.get_idle_workers()
                    if len(workers) == 0:
                        workers = self.get_harvester_workers()
                    assert position is not None, "The next supply depot position is None"
                    assert len(workers) > 0, "There are no workers to build the supply depot"
                    worker, _ = self._get_closest(workers, position)
                    action_args = dict(source_unit_tag=worker.tag, target_position=position)
                case AllActions.BUILD_COMMAND_CENTER:
                    position = self.get_next_command_center_position()
                    workers = self.get_idle_workers()
                    if len(workers) == 0:
                        workers = self.get_harvester_workers()
                    assert position is not None, "The next command center position is None"
                    assert len(workers) > 0, "There are no workers to build the command center"
                    worker, _ = self._get_closest(workers, position)
                    action_args = dict(source_unit_tag=worker.tag, target_position=position)
                case AllActions.BUILD_BARRACKS:
                    position = self.get_next_barracks_position()
                    workers = self.get_idle_workers()
                    if len(workers) == 0:
                        workers = self.get_harvester_workers()
                    assert position is not None, "The next barracks position is None"
                    assert len(workers) > 0, "There are no workers to build the barracks"
                    worker, _ = self._get_closest(workers, position)
                    action_args = dict(source_unit_tag=worker.tag, target_position=position)
                case AllActions.RECRUIT_MARINE:
                    barracks = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.Barracks, completed_only=True)
                    barracks = [cc for cc in barracks if cc.order_length < Constants.BARRACKS_QUEUE_LENGTH]
                    assert len(barracks) > 0, "There are no barracks available"
                    barracks = sorted(barracks, key=lambda b: b.order_length)
                    barrack = barracks[0]
                    action_args = dict(source_unit_tag=barrack.tag)
                case AllActions.ATTACK_CLOSEST_BUILDING:
                    marines = self.get_entire_army()
                    assert len(marines) > 0, "There are no marines"
                    enemies = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.BUILDING_UNIT_TYPES)
                    assert len(enemies) > 0, "There are no enemy buildings"
                    marines_avg_position = self._get_mean_position(marines)
                    target, _ = self._get_closest(enemies, marines_avg_position)
                    marine_tags = [m.tag for m in marines]
                    action_args = dict(source_unit_tag=marine_tags, target_unit_tag=target.tag)
                case AllActions.ATTACK_CLOSEST_WORKER:
                    marines = self.get_entire_army()
                    assert len(marines) > 0, "There are no marines"
                    enemies = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.WORKER_UNIT_TYPES)
                    assert len(enemies) > 0, "There are no enemy workers"
                    marines_avg_position = self._get_mean_position(marines)
                    target, _ = self._get_closest(enemies, marines_avg_position)
                    marine_tags = [m.tag for m in marines]
                    action_args = dict(source_unit_tag=marine_tags, target_unit_tag=target.tag)
                case AllActions.ATTACK_CLOSEST_ARMY:
                    marines = self.get_entire_army()
                    assert len(marines) > 0, "There are no marines"
                    enemies = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.ARMY_UNIT_TYPES)
                    assert len(enemies) > 0, "There are no enemy units"
                    marines_avg_position = self._get_mean_position(marines)
                    target, _ = self._get_closest(enemies, marines_avg_position)
                    marine_tags = [m.tag for m in marines]
                    action_args = dict(source_unit_tag=marine_tags, target_unit_tag=target.tag)
                case AllActions.ATTACK_BUILDINGS:
                    marines = self.get_entire_army()
                    assert len(marines) > 0, "There are no marines"
                    enemies = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.BUILDING_UNIT_TYPES)
                    assert len(enemies) > 0, "There are no enemy buildings"
                    enemies_avg_position = self._get_mean_position(enemies)
                    target, _ = self._get_closest(enemies, enemies_avg_position)
                    marine_tags = [m.tag for m in marines]
                    action_args = dict(source_unit_tag=marine_tags, target_unit_tag=target.tag)
                case AllActions.ATTACK_WORKERS:
                    marines = self.get_entire_army()
                    assert len(marines) > 0, "There are no marines"
                    enemies = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.WORKER_UNIT_TYPES)
                    assert len(enemies) > 0, "There are no enemy workers"
                    enemies_avg_position = self._get_mean_position(enemies)
                    target, _ = self._get_closest(enemies, enemies_avg_position)
                    marine_tags = [m.tag for m in marines]
                    action_args = dict(source_unit_tag=marine_tags, target_unit_tag=target.tag)
                case AllActions.ATTACK_ARMY:
                    marines = self.get_entire_army()
                    assert len(marines) > 0, "There are no marines"
                    enemies = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.ARMY_UNIT_TYPES)
                    assert len(enemies) > 0, "There are no enemy units"
                    enemies_avg_position = self._get_mean_position(enemies)
                    target, _ = self._get_closest(enemies, enemies_avg_position)
                    marine_tags = [m.tag for m in marines]
                    action_args = dict(source_unit_tag=marine_tags, target_unit_tag=target.tag)
                case _:
                    raise RuntimeError(f"Missing logic to select action args for action {action.name}")

            return action_args, True
        except AssertionError as error:
            self.logger.debug(error)
            return None, False

    def get_next_command_center_position(self) -> Position:
        next_pos = None
        command_centers = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.CommandCenter)
        command_centers_positions = [Position(cc.x, cc.y) for cc in command_centers]

        enemy_command_centers = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.COMMAND_CENTER_UNIT_TYPES)
        enemy_command_centers_positions = [Position(cc.x, cc.y) for cc in enemy_command_centers]
        all_cc_positions = command_centers_positions + enemy_command_centers_positions
        for idx, candidate_position in enumerate(self._command_center_positions):
            if (candidate_position not in self._used_command_center_positions) and (candidate_position not in all_cc_positions):
                next_pos = candidate_position
                break

        return next_pos

    def update_command_center_positions(self):
        command_centers = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.CommandCenter)
        command_center_positions = [Position(cc.x, cc.y) for cc in command_centers]
        enemy_command_centers = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.COMMAND_CENTER_UNIT_TYPES)
        enemy_command_center_positions = [Position(cc.x, cc.y) for cc in enemy_command_centers]
        self._used_command_center_positions = command_center_positions + enemy_command_center_positions

    def get_next_supply_depot_position(self) -> Position:
        next_pos = None
        for idx, candidate_position in enumerate(self._supply_depot_positions):
            # due to rounding in min-games, the supply depots are placed one unit too far to the right
            # so that position needs to be checked as well
            displaced_candidate_position = Position(candidate_position.x+1, candidate_position.y)
            if candidate_position not in self._used_supply_depot_positions\
                and displaced_candidate_position not in self._used_supply_depot_positions:
                next_pos = candidate_position
                break

        return next_pos

    def update_supply_depot_positions(self):
        supply_depots = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.SupplyDepot)
        supply_depots_positions = [Position(sd.x, sd.y) for sd in supply_depots]
        self._used_supply_depot_positions = supply_depots_positions

    def get_next_barracks_position(self) -> Position:
        next_pos = None
        for idx, candidate_position in enumerate(self._barrack_positions):
            if candidate_position not in self._used_barrack_positions:
                next_pos = candidate_position
                break

        return next_pos

    def update_barracks_positions(self):
        barracks = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.Barracks)
        barrack_positions = [Position(b.x, b.y) for b in barracks]
        self._used_barrack_positions = barrack_positions

    def _get_mean_position(self, units):
        if units is None or len(units) == 0:
            return None
        unit_positions = [Position(e.x, e.y) for e in units]
        mean_unit_position = np.mean(unit_positions, axis=0)
        return mean_unit_position

    def _get_mean_unit(self, units):
        mean_unit_position = self._get_mean_position(units)
        if mean_unit_position is None:
            return None
        target_position = Position(int(mean_unit_position[0]), int(mean_unit_position[1]))
        mean_unit, _ = self._get_closest(units, target_position)
        return mean_unit

    def _get_distances(self, units: List[features.FeatureUnit], position: Position) -> List[float]:
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(position), axis=1)

    def _get_distance(self, pos1: Position, pos2: Position):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def _get_closest(self, units: List[features.FeatureUnit], position: Position) -> Tuple[features.FeatureUnit, float]:
        if position is None or units is None or len(units) == 0:
            return None, None
        distances = self._get_distances(units, position)
        min_distance = distances.min()
        min_distances = np.where(distances == min_distance)[0]
        # If there is only one minimum distance, that will be returned, otherwise we return one of the elements with the minimum distance

        closes_unit_idx = np.random.choice(min_distances)
        return units[closes_unit_idx], min_distance

    def get_reward_and_score(self, obs: TimeStep, eval_step: bool = True) -> Tuple[float, float, float]:
        reward = obs.reward

        get_score = getattr(self, self._score_method)
        score = get_score(obs)
        adjusted_reward = reward + Constants.STEP_REWARD

        if not (obs.last() or obs.first()):
            if not self._prev_action_is_valid:
                adjusted_reward = Constants.INVALID_ACTION_REWARD
            # elif self._prev_action == AllActions.NO_OP:
            #     adjusted_reward = Constants.NO_OP_REWARD

        self._current_score = score
        self._current_reward = reward
        self._current_adjusted_reward = adjusted_reward

        if not eval_step:
            return 0, 0, 0

        return reward, adjusted_reward, score

    def get_army_health_difference(self, obs: TimeStep) -> float:
        enemy_army = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.ARMY_UNIT_TYPES)
        enemy_total_army_health = sum(map(lambda b: b.health, enemy_army))
        marines = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.Marine)
        total_army_health = sum(map(lambda b: b.health, marines))

        return total_army_health - enemy_total_army_health

    def get_mineral_count_delta(self, obs: TimeStep) -> float:
        minerals = obs.observation.player.minerals
        if obs.first():
            self._prev_minerals = minerals
            return 0
        else:
            prev = self._prev_minerals
            delta = minerals - prev
            self._prev_minerals = minerals
            return delta

    def get_army_spending(self) -> float:
        barracks = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.Barracks)
        # depots = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.SupplyDepot)
        marines = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.Marine)
        n_marines_in_queue = sum([b.order_length for b in barracks])

        army_spending = SC2Costs.MARINE.minerals * (len(marines) + n_marines_in_queue) # marines
        # econ_spending = SC2Costs.SUPPLY_DEPOT.minerals * len(depots) # command centers, supply depots, scvs
        # tech_spending = SC2Costs.BARRACKS.minerals * len(barracks) # barracks

        # return army_spending - (econ_spending + tech_spending)
        return army_spending

    def get_army_spending_delta(self, obs: TimeStep) -> float:
        spending = self.get_army_spending()
        if obs.first():
            self._prev_army_spending = spending
            return 0
        else:
            prev = self._prev_army_spending
            delta = spending - prev
            self._prev_army_spending = spending
            return delta

    def get_diff_marines(self) -> float:
        marines = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.Marine)
        enemy_marines = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=units.Terran.Marine)
        diff_marines = len(marines) - len(enemy_marines)

        return diff_marines

    def get_diff_marines_delta(self, obs: TimeStep) -> float:
        diff_marines = self.get_diff_marines()
        if obs.first() or obs.last(): # ignore last step since the game seems to delete all units, so the difference becomes 0 and that can cause a large reward or penalty
            self._prev_diff_marines = diff_marines
            return 0
        else:
            prev = self._prev_diff_marines
            delta = diff_marines - prev
            self._prev_diff_marines = diff_marines
            return delta

    def get_health_difference_score(self) -> float:
        enemy_buildings = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.BUILDING_UNIT_TYPES)
        buildings_health = sum([u.health for u in enemy_buildings])
        enemy_workers = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.WORKER_UNIT_TYPES)
        workers_health = sum([u.health for u in enemy_workers])
        enemy_army = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.ARMY_UNIT_TYPES)
        army_health = sum([u.health for u in enemy_army])
        marines = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.Marine)
        marines_health = sum([u.health for u in marines])

        return marines_health - (buildings_health + workers_health + army_health)

    def get_health_difference_score_delta(self, obs: TimeStep) -> float:
        factor = 0.1
        step_cost = 1
        health_difference_score = self.get_health_difference_score()
        if obs.first():
            self._prev_health_difference_score = health_difference_score
            return 0
        else:
            prev = self._prev_health_difference_score
            delta = health_difference_score - prev
            self._prev_health_difference_score = health_difference_score
            return delta * factor - step_cost

    def get_reward_as_score(self, obs: TimeStep) -> float:
        return obs.reward

    def get_game_score(self) -> float:
        damaged_unit_min_factor = 0.5
        unit_in_progress_factor = 0.25

        def get_units_value(units, unit_cost: SC2Costs):
            return sum([(max(damaged_unit_min_factor, u.health_ratio / 255) * unit_cost.minerals) if self.is_complete(u)
                        else (unit_in_progress_factor * unit_cost.minerals)
                        for u in units])

        def get_player_score(player: PlayerRelative):
            command_centers = self._get_units(alliances=player, unit_types=units.Terran.CommandCenter)
            supply_depots = self._get_units(alliances=player, unit_types=units.Terran.SupplyDepot)
            barracks = self._get_units(alliances=player, unit_types=units.Terran.Barracks)
            marines = self._get_units(alliances=player, unit_types=units.Terran.Marine)
            workers = self._get_units(alliances=player, unit_types=units.Terran.SCV)
            marines_in_progress = sum([cc.order_length for cc in command_centers])
            workers_in_progress = sum([b.order_length for b in barracks])

            return get_units_value(command_centers, SC2Costs.COMMAND_CENTER) \
                    + get_units_value(supply_depots, SC2Costs.SUPPLY_DEPOT) \
                    + get_units_value(barracks, SC2Costs.BARRACKS) \
                    + get_units_value(marines, SC2Costs.MARINE) \
                    + marines_in_progress * unit_in_progress_factor \
                    + get_units_value(workers, SC2Costs.SCV) \
                    + workers_in_progress * unit_in_progress_factor

        ally_score = get_player_score(PlayerRelative.SELF)
        enemy_score = get_player_score(PlayerRelative.ENEMY)

        return ally_score - enemy_score


    def get_game_score_delta(self, obs: TimeStep) -> float:
        win_factor = 1000
        step_cost = 5
        game_score = self.get_game_score()
        if obs.first():
            self._prev_game_score = 0
            return 0
        elif obs.last():
            self._prev_game_score = game_score
            return win_factor * obs.reward
        else:
            prev = self._prev_game_score
            delta = game_score - prev
            self._prev_game_score = game_score
            return delta - step_cost

    def _convert_obs_to_state(self, obs: TimeStep) -> Tuple:
        actions_state = self._get_actions_state()
        building_state = self._get_buildings_state()
        worker_state = self._get_workers_state()
        army_state = self._get_army_state()
        resources_state = self._get_resources_state(obs)
        scores_state = self._get_scores_state(obs)
        neutral_units_state = self._get_neutral_units_state()
        enemy_state = self._get_enemy_state()
        # Enemy

        state_tuple = State(
            **actions_state,
			**building_state,
			**worker_state,
			**army_state,
			**resources_state,
            **scores_state,
            **neutral_units_state,
            **enemy_state
        )
        return state_tuple

    def _actions_to_network(self, actions: List[AllActions]) -> List:
        """Converts a list of AllAction elements to a one-hot encoded version that the network can use.

        Args:
            actions (List[AllActions]): List of actions

        Returns:
            Tensor: One-hot encoded version of the actions provided.
        """
        ohe_actions = np.zeros(self._num_actions, dtype=np.int8)

        for action in actions:
            action_idx = self._action_to_idx[action]
            ohe_actions[action_idx] = 1

        return ohe_actions

    def pre_step(self, obs: TimeStep, eval_step: bool = True):
        self._current_obs_unit_info = self._gather_obs_info(obs)

        self.update_supply_depot_positions()
        self.update_command_center_positions()
        self.update_barracks_positions()

        self._available_actions = self.calculate_available_actions(obs)
        self._current_state_tuple = self._convert_obs_to_state(obs)

        if not eval_step:
            self.setup_actions()
        else:
            reward, adjusted_reward, score = self.get_reward_and_score(obs, eval_step)
            self._current_episode_stats.reward += reward
            self._current_episode_stats.adjusted_reward += adjusted_reward
            self._current_episode_stats.score += score
            self._current_episode_stats.steps += 1
            self.current_agent_stats.step_count += 1
            a = {RewardMode.SCORE: ("score", score), RewardMode.ADJUSTED_REWARD: ("adjusted reward", adjusted_reward), RewardMode.REWARD: ("reward", reward)}
            self.logger.debug(f"Previous action {a[self._reward_mode][0]}: {a[self._reward_mode][1]}")

    def post_step(self, obs: TimeStep, action: AllActions, action_args: Dict[str, Any], original_action: AllActions, original_action_args: Dict[str, Any], is_valid_action: bool):
        self._prev_score = obs.observation.score_cumulative.score
        self._prev_state_tuple = self._current_state_tuple
        self._prev_action = self._action_to_idx[original_action]
        self._prev_action_args = original_action_args
        self._prev_action_is_valid = is_valid_action

        if is_valid_action:
            self._current_episode_stats.add_valid_action(action)
        else:
            self._current_episode_stats.add_invalid_action(action)

        # if not self._exploit:
        if obs.last():
            self._store_episode_stats()
        else:
            now = time.time()
            if (self._tracker is not None) and (now - self._tracker_last_update > self._tracker_update_freq_seconds):
                emissions = self._tracker.flush() or 0.
                self.logger.debug(f"Tracker flush - Got extra {emissions} since last update")
                self._current_episode_stats.emissions += emissions
                self._tracker_last_update = now

        self._current_obs_unit_info = None

    def _store_episode_stats(self):
        emissions = self._tracker.flush() if self._tracker is not None else 0.
        self.logger.debug(f"End of episode - Got extra {emissions} since last update")
        self._current_episode_stats.emissions += emissions
        self._current_episode_stats.is_training = self.is_training
        self._current_episode_stats.is_exploit = self._exploit
        self._current_episode_stats.final_stage = self._current_agent_stage().name
        self._tracker_last_update = time.time()
        episode_stage = self._current_episode_stats.initial_stage
        mean_rewards = self.current_aggregated_episode_stats.mean_rewards(stage=episode_stage,
                                                                          reward_mode=self._reward_mode, last_n=5)
        max_mean_rewards = self._best_mean_rewards
        max_mean_rewards_str = f"{max_mean_rewards:.2f}" if max_mean_rewards is not None else "None"
        new_max_mean_rewards = (max_mean_rewards is None) or (mean_rewards >= max_mean_rewards)

        self.current_agent_stats.process_episode(self._current_episode_stats)
        self.current_aggregated_episode_stats.process_episode(self._current_episode_stats)
        self._episode_stats[self._map_name].append(self._current_episode_stats)
        log_msg_parts = ["\n=================", "================="] + self._get_end_of_episode_info_components() + [
            "=================", "================="]
        log_msg = "\n".join(log_msg_parts)
        self.logger.info(log_msg)

    def _get_end_of_episode_info_components(self) -> List[str]:
        num_invalid = sum(self._current_episode_stats.invalid_action_counts.values())
        num_valid = sum(self._current_episode_stats.valid_action_counts.values())
        pct_invalid = num_invalid / (num_invalid + num_valid) if num_invalid + num_valid > 0 else 0
        episode_stage = self._current_episode_stats.initial_stage
        mean_rewards = self.current_aggregated_episode_stats.mean_rewards(stage=episode_stage, reward_mode=RewardMode.REWARD)
        mean_rewards_10 = self.current_aggregated_episode_stats.mean_rewards(stage=episode_stage, last_n=10, reward_mode=RewardMode.REWARD)
        mean_adjusted_rewards = self.current_aggregated_episode_stats.mean_rewards(stage=episode_stage, reward_mode=RewardMode.ADJUSTED_REWARD)
        mean_adjusted_rewards_10 = self.current_aggregated_episode_stats.mean_rewards(stage=episode_stage, last_n=10, reward_mode=RewardMode.ADJUSTED_REWARD)
        mean_scores = self.current_aggregated_episode_stats.mean_rewards(stage=episode_stage, reward_mode=RewardMode.SCORE)
        mean_scores_10 = self.current_aggregated_episode_stats.mean_rewards(stage=episode_stage, last_n=10, reward_mode=RewardMode.SCORE)
        episode_count = self.current_agent_stats.episode_count_per_stage[episode_stage]
        return [
            f"Episode {self._map_name} // Stage: {episode_stage} // Final stage: {self._current_agent_stage().name}",
            f"Reward method: {self._reward_mode.name}",
            f"Reward: {self._current_episode_stats.reward} // Adj. Reward: {self._current_episode_stats.adjusted_reward} // Score: {self._current_episode_stats.score}",
            f"Episode {episode_count}",
            f"Mean Rewards for stage ({episode_count} ep) {mean_rewards:.2f} / (10ep) {mean_rewards_10:.2f}",
            f"Mean Adjusted Rewards for stage ({episode_count} ep) {mean_adjusted_rewards:.2f} / (10ep) {mean_adjusted_rewards_10:.2f}",
            f"Mean Scores ({episode_count} ep) {mean_scores:.2f} / (10ep) {mean_scores_10:.2f}",
            f"Episode steps: {self._current_episode_stats.steps} / Total steps: {self.current_agent_stats.step_count_per_stage[episode_stage]}",
            f"Invalid action masking: {self._action_masking}",
            f"Invalid actions: {num_invalid}/{num_valid + num_invalid} ({100 * pct_invalid:.2f}%)",
            f"Max reward {self.current_aggregated_episode_stats.max_reward_per_stage[episode_stage]:.2f} (absolute max: {self.current_aggregated_episode_stats.max_reward})",
            f"Max adjusted reward {self.current_aggregated_episode_stats.max_adjusted_reward_per_stage[episode_stage]:.2f} (absolute max: {self.current_aggregated_episode_stats.max_adjusted_reward})",
            f"Max score {self.current_aggregated_episode_stats.max_score_per_stage[episode_stage]:.2f} (absolute max: {self.current_aggregated_episode_stats.max_score})",
            f"Episode emissions: {self._current_episode_stats.emissions} / Total in stage: {self.current_agent_stats.total_emissions_per_stage[episode_stage]} / Total: {self.current_agent_stats.total_emissions}",
        ]

    def step(self, obs: TimeStep, only_super_step: bool = False) -> AllActions:
        if only_super_step:
            super().step(obs)
            return

        if obs.first():
            self.setup_positions(obs)

        self.pre_step(obs, not obs.first())

        super().step(obs)

        action, action_args, is_valid_action = self.select_action(obs)
        original_action = action
        original_action_args = action_args

        if not is_valid_action:
            self.logger.debug(f"Action {action.name} is not valid anymore, returning NO_OP")
            action = AllActions.NO_OP
            action_args = None

        if is_valid_action:
            self.logger.debug(f"[Step {self.steps}] Performing action {action.name} with args: {action_args}")

        game_action = self._action_to_game[action]

        self.post_step(obs, action, action_args, original_action, original_action_args, is_valid_action)

        if action_args is not None:
            return game_action(**action_args)

        return game_action()

    def calculate_available_actions(self, obs: TimeStep) -> List[AllActions]:
        can_take = {}

        completed_command_centers = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.CommandCenter, completed_only=True)
        cc0_can_exist = self._command_center_0_pos is not None
        cc1_can_exist = self._command_center_1_pos is not None
        cc2_can_exist = self._command_center_2_pos is not None
        command_center_0 = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.CommandCenter,
                                         positions=[self._command_center_0_pos], completed_only=True, first_only=True)
        cc0_queue_is_full = command_center_0.order_length >= Constants.COMMAND_CENTER_QUEUE_LENGTH if command_center_0 is not None else True
        command_center_1 = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.CommandCenter,
                                         positions=[self._command_center_1_pos], completed_only=True, first_only=True)
        cc1_queue_is_full = command_center_1.order_length >= Constants.COMMAND_CENTER_QUEUE_LENGTH if command_center_1 is not None else True
        command_center_2 = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.CommandCenter,
                                         positions=[self._command_center_2_pos], completed_only=True, first_only=True)
        cc2_queue_is_full = command_center_2.order_length >= Constants.COMMAND_CENTER_QUEUE_LENGTH if command_center_2 is not None else True

        completed_supply_depots = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.SupplyDepot, completed_only=True)
        completed_barracks = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.Barracks, completed_only=True)
        barracks_can_queue = any([b.order_length < Constants.BARRACKS_QUEUE_LENGTH for b in completed_barracks])

        minerals = [unit for unit in self._get_units(alliances=PlayerRelative.NEUTRAL) if
                    Minerals.contains(unit.unit_type)]

        has_idle_workers = self.has_idle_workers()
        has_harvester_workers = self.has_harvester_workers()
        has_marines = self.has_marines()

        can_pay_scv = SC2Costs.SCV.can_pay(obs.observation.player)
        can_pay_marine = SC2Costs.MARINE.can_pay(obs.observation.player)
        can_pay_supply_depot = SC2Costs.SUPPLY_DEPOT.can_pay(obs.observation.player)
        can_pay_command_center = SC2Costs.COMMAND_CENTER.can_pay(obs.observation.player)
        can_pay_barrack = SC2Costs.BARRACKS.can_pay(obs.observation.player)

        next_supply_depot_position = self.get_next_supply_depot_position()
        next_command_center_position = self.get_next_command_center_position()
        next_barracks_position = self.get_next_barracks_position()

        enemy_buildings = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.BUILDING_UNIT_TYPES)
        enemy_workers = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.WORKER_UNIT_TYPES)
        enemy_army = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.ARMY_UNIT_TYPES)

        can_take[AllActions.NO_OP] = True
        can_take[AllActions.HARVEST_MINERALS] = len(completed_command_centers) > 0 and len(minerals) > 0 and has_idle_workers
        can_take[AllActions.RECRUIT_SCV_0] = cc0_can_exist and can_pay_scv and command_center_0 is not None and not cc0_queue_is_full
        can_take[AllActions.RECRUIT_SCV_1] = cc1_can_exist and can_pay_scv and command_center_1 is not None and not cc1_queue_is_full
        can_take[AllActions.RECRUIT_SCV_2] = cc2_can_exist and can_pay_scv and command_center_2 is not None and not cc2_queue_is_full
        can_take[AllActions.BUILD_SUPPLY_DEPOT] = next_supply_depot_position is not None and can_pay_supply_depot and (has_idle_workers or has_harvester_workers)
        can_take[AllActions.BUILD_COMMAND_CENTER] = next_command_center_position is not None and can_pay_command_center and (has_idle_workers or has_harvester_workers)
        can_take[AllActions.BUILD_BARRACKS] = next_barracks_position is not None and can_pay_barrack and (has_idle_workers or has_harvester_workers) and len(completed_supply_depots) > 0
        can_take[AllActions.RECRUIT_MARINE] = can_pay_marine and len(completed_barracks) > 0 and barracks_can_queue
        can_take[AllActions.ATTACK_CLOSEST_BUILDING] = has_marines and len(enemy_buildings) > 0
        can_take[AllActions.ATTACK_CLOSEST_WORKER] = has_marines and len(enemy_workers) > 0
        can_take[AllActions.ATTACK_CLOSEST_ARMY] = has_marines and len(enemy_army) > 0
        can_take[AllActions.ATTACK_BUILDINGS] = has_marines and len(enemy_buildings) > 0
        can_take[AllActions.ATTACK_WORKERS] = has_marines and len(enemy_workers) > 0
        can_take[AllActions.ATTACK_ARMY] = has_marines and len(enemy_army) > 0

        available_actions = [a for a in self.agent_actions if a in self._map_config["available_actions"] and can_take[a]]

        return available_actions

    def get_idle_marines(self) -> List[features.FeatureUnit]:
        """Gets all idle workers.

        Args:
            obs (TimeStep): Observation from the environment

        Returns:
            List[features.FeatureUnit]: List of idle workers
        """
        self_marines = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.Marine)
        idle_marines = filter(self.is_idle, self_marines)

        return list(idle_marines)

    def get_entire_army(self) -> List[features.FeatureUnit]:
        return self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.Marine)

    def has_marines(self) -> bool:
        return len(self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.Marine)) > 0

    def has_idle_workers(self) -> bool:
        return len(self.get_idle_workers()) > 0

    def has_harvester_workers(self) -> bool:
        return len(self.get_harvester_workers()) > 0

    def has_workers(self) -> bool:
        return len(self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.SCV)) > 0

    def _gather_obs_info(self, obs: TimeStep) -> Dict:
        unit_info = {}
        for unit in obs.observation.raw_units:
            if unit.alliance not in unit_info:
                unit_info[unit.alliance] = {}

            if unit.unit_type not in unit_info[unit.alliance]:
                unit_info[unit.alliance][unit.unit_type] = []

            unit_info[unit.alliance][unit.unit_type].append(unit)

        return unit_info

    def _get_units(self, alliances: PlayerRelative | List[PlayerRelative] = None,  unit_types: int | List[int] = None, unit_tags: int | List[int] = None, positions: Position | List[Position] = None, completed_only: bool = False, first_only: bool = False) -> features.FeatureUnit | List[features.FeatureUnit] | None:
        """
        Creates a nested dictionary with two levels containing all units in the game.
        The first level groups de units by player and the second level groups them by unit type.
        """
        if self._current_obs_unit_info is None:
            raise RuntimeError("Unit info is not initialized")

        if alliances is None:
            alliances = list(self._current_obs_unit_info.keys())
        else:
            alliances = [alliances] if isinstance(alliances, PlayerRelative) else alliances
        type_dicts = [self._current_obs_unit_info[a] for a in alliances if a in self._current_obs_unit_info]

        if unit_types is None:
            unit_types = list(set(key for dict in type_dicts for key in list(dict.keys())))
        else:
            unit_types = [unit_types] if isinstance(unit_types, int) else unit_types
        unit_list = [unit for dict in type_dicts for type in unit_types if type in dict for unit in dict[type]]

        if unit_tags is not None:
            unit_tags = [unit_tags] if isinstance(unit_tags, (int, np.int64, np.integer, np.int32)) else unit_tags
            tags_lambda = lambda unit: unit.tag in unit_tags
        else:
            tags_lambda = lambda unit: True

        if positions is not None:
            positions = [positions] if isinstance(positions, Position) else positions
            positions_lambda = lambda unit: Position(unit.x, unit.y) in positions
        else:
            positions_lambda = lambda unit: True

        if completed_only:
            completed_lambda = lambda unit: unit.build_progress == 100
        else:
            completed_lambda = lambda unit: True

        filter_lambda = lambda unit: tags_lambda(unit) and positions_lambda(unit) and completed_lambda(unit)

        filtered_units = list(filter(filter_lambda , unit_list))

        if first_only:
            if len(filtered_units) > 0:
                return filtered_units[0]
            else:
                return None
        else:
            return filtered_units

    @deprecated("use self._get_units() instead")
    def get_self_units(self, obs: TimeStep, unit_types: int | List[int] = None, unit_tags: int | List[int] = None, positions: Position | List[Position] = None, completed_only: bool = False, first_only: bool = False) -> features.FeatureUnit|List[features.FeatureUnit]|None:
        """Get a list of the player's own units.

        Args:
            obs (TimeStep): Observation from the environment
            unit_type (int | List[int], optional): Type of unit(s) to get. If provided, only units of this type(s) will be
                                       returned, otherwise all units are returned.

        Returns:
            List[features.FeatureUnit]: List of player units, filtered by unit type and/or tag if provided
        """
        units = filter(lambda u: u.alliance == PlayerRelative.SELF, obs.observation.raw_units)

        if unit_types is not None:
            unit_types = [unit_types] if isinstance(unit_types, int) else unit_types
            units = filter(lambda u: u.unit_type in unit_types, units)

        if unit_tags is not None:
            unit_tags = [unit_tags] if isinstance(unit_tags, (int, np.int64, np.integer, np.int32)) else unit_tags
            units = filter(lambda u: u.tag in unit_tags, units)

        if positions is not None:
            positions = [positions] if isinstance(positions, Position) else positions
            units = filter(lambda u: Position(u.x, u.y) in positions, units)

        if completed_only:
            units = filter(lambda u: u.build_progress == 100, units)

        unit_list = list(units)
        if first_only:
            if len(unit_list) > 0:
                return unit_list[0]
            else:
                return None
        else:
            return unit_list

    @deprecated("use self._get_units() instead")
    def get_neutral_units(self, obs: TimeStep, unit_types: int | List[int] = None, unit_tags: int | List[int] = None) -> List[features.FeatureUnit]:
        """Get a list of neutral units.

        Args:
            obs (TimeStep): Observation from the environment
            unit_type (int | List[int], optional): Type of unit(s) to get. If provided, only units of this type(s) will be
                                       returned, otherwise all units are returned.

        Returns:
            List[features.FeatureUnit]: List of neutral units, filtered by unit type and/or tag if provided
        """
        units = filter(lambda u: u.alliance == PlayerRelative.ENEMY, obs.observation.raw_units)

        if unit_types is not None:
            unit_types = [unit_types] if isinstance(unit_types, int) else unit_types
            units = filter(lambda u: u.unit_type in unit_types, units)

        if unit_tags is not None:
            unit_tags = [unit_tags] if isinstance(unit_tags, (int, np.int64, np.integer, np.int32)) else unit_tags
            units = filter(lambda u: u.tag in unit_tags, units)

        return list(units)

    @deprecated("use self._get_units() instead")
    def get_enemy_units(self, obs: TimeStep, unit_types: int | List[int] = None, unit_tags: int | List[int] = None) -> List[features.FeatureUnit]:
        """Get a list of the player's enemy units.

        Args:
            obs (TimeStep): Observation from the environment
            unit_type (int | List[int], optional): Type of unit(s) to get. If provided, only units of this type(s) will be
                                       returned, otherwise all units are returned.

        Returns:
            List[features.FeatureUnit]: List of enemy units, filtered by unit type and/or tag if provided
        """
        units = filter(lambda u: u.alliance == PlayerRelative.ENEMY, obs.observation.raw_units)

        if unit_types is not None:
            unit_types = [unit_types] if isinstance(unit_types, int) else unit_types
            units = filter(lambda u: u.unit_type in unit_types, units)

        if unit_tags is not None:
            unit_tags = [unit_tags] if isinstance(unit_tags, (int, np.int64, np.integer, np.int32)) else unit_tags
            units = filter(lambda u: u.tag in unit_tags, units)

        return list(units)

    def get_workers(self, idle: bool = False, harvesting: bool = False) -> List[features.FeatureUnit]:
        """Gets all idle workers.

        Args:
            obs (TimeStep): Observation from the environment

        Returns:
            List[features.FeatureUnit]: List of idle workers
        """
        if idle and harvesting:
            self.logger.warning(f"Asking for workers that are idle AND harvesting will always result in an empty list")
            return []

        workers = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.SCV)

        if idle:
            workers = filter(self.is_idle, workers)
        elif harvesting:
            workers = filter(lambda w: w.order_id_0 in Constants.HARVEST_ACTIONS, workers)

        return list(workers)

    def get_idle_workers(self) -> List[features.FeatureUnit]:
        """Gets all idle workers.

        Args:
            obs (TimeStep): Observation from the environment

        Returns:
            List[features.FeatureUnit]: List of idle workers
        """
        return self.get_workers(idle=True)

    def get_harvester_workers(self) -> List[features.FeatureUnit]:
        """Get a list of all workers that are currently harvesting.

        Args:
            obs (TimeStep): Observation from the environment.

        Returns:
            List[features.FeatureUnit]: List of workers that are harvesting.
        """
        return self.get_workers(harvesting=True)

    def get_free_supply(self, obs: TimeStep) -> int:
        return obs.observation.player.food_cap - obs.observation.player.food_used

    def is_idle(self, unit: features.FeatureUnit) -> bool:
        """Check whether a unit is idle (meaning it has no orders in the queue)"""
        return unit.order_length == 0

    def is_complete(self, unit: features.FeatureUnit) -> bool:
        """Check whether a unit is fully build"""
        return unit.build_progress == 100

    def _get_buildings_state(self):
        def _get_complete(buildings):
            return list(filter(self.is_complete, buildings))

        # info about command centers
        command_centers = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.CommandCenter)
        num_command_centers = len(command_centers)
        max_command_centers = len(self._command_center_positions)
        completed_command_centers = _get_complete(command_centers)
        num_completed_command_centers = len(completed_command_centers)
        pct_command_centers = 1 if max_command_centers == 0 else num_command_centers / max_command_centers
        command_centers_state = dict(
            num_command_centers=num_command_centers,
            num_completed_command_centers=num_completed_command_centers,
            max_command_centers=max_command_centers,
            pct_command_centers=pct_command_centers,
        )

        command_center_0 = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.CommandCenter, positions=[self._command_center_0_pos], first_only=True)\
            if self._command_center_0_pos is not None else None
        command_center_1 = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.CommandCenter, positions=[self._command_center_1_pos], first_only=True)\
            if self._command_center_1_pos is not None else None
        command_center_2 = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.CommandCenter, positions=[self._command_center_2_pos], first_only=True)\
            if self._command_center_2_pos is not None else None

        command_centers_state[f"command_center_0_order_length"] = command_center_0.order_length if command_center_0 is not None else -1
        command_centers_state[f"command_center_0_num_workers"] = command_center_0.assigned_harvesters if command_center_0 is not None else -1

        command_centers_state[f"command_center_1_order_length"] = command_center_1.order_length if command_center_1 is not None else -1
        command_centers_state[f"command_center_1_num_workers"] = command_center_1.assigned_harvesters if command_center_1 is not None else -1

        command_centers_state[f"command_center_2_order_length"] = command_center_2.order_length if command_center_2 is not None else -1
        command_centers_state[f"command_center_2_num_workers"] = command_center_2.assigned_harvesters if command_center_2 is not None else -1

        # Buildings
        supply_depots = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.SupplyDepot)
        num_supply_depots = len(supply_depots)
        completed_supply_depots = _get_complete(supply_depots)
        num_completed_supply_depots = len(completed_supply_depots)
        max_supply_depots = len(self._supply_depot_positions)
        pct_supply_depots = 1 if max_supply_depots == 0 else num_supply_depots / max_supply_depots

        barracks = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.Barracks)
        num_barracks = len(barracks)
        completed_barracks = _get_complete(barracks)
        num_completed_barracks = len(completed_barracks)
        max_barracks = len(self._barrack_positions)
        pct_barracks = 1 if max_barracks == 0 else num_barracks / max_barracks
        barracks_total_queue_length = len(completed_barracks) * Constants.BARRACKS_QUEUE_LENGTH if num_completed_barracks > 0 else -1
        barracks_used_queue_length = sum([b.order_length for b in completed_barracks]) if num_completed_barracks > 0 else -1
        barracks_free_queue_length = barracks_total_queue_length - barracks_used_queue_length if num_completed_barracks > 0 else -1

        buildings_state = dict(
			# Supply Depots
			num_supply_depots=num_supply_depots,
			num_completed_supply_depots=num_completed_supply_depots,
			max_supply_depots=max_supply_depots,
			pct_supply_depots=pct_supply_depots,
			# Barracks
			num_barracks=num_barracks,
			num_completed_barracks=num_completed_barracks,
			max_barracks=max_barracks,
			pct_barracks=pct_barracks,
			barracks_used_queue_length=barracks_used_queue_length,
			barracks_free_queue_length=barracks_free_queue_length,
        )

        return {
            **command_centers_state,
            **buildings_state
        }

    def _get_workers_state(self) -> Dict[str, int|float]:
        workers = self.get_workers()
        num_mineral_harvesters = len([w for w in workers if w.order_id_0 in Constants.HARVEST_ACTIONS])
        num_workers = len(workers)
        num_idle_workers = len([w for w in workers if self.is_idle(w)])
        pct_idle_workers = 0 if num_workers == 0 else num_idle_workers / num_workers
        pct_mineral_harvesters = 0 if num_workers == 0 else num_mineral_harvesters / num_workers

        # TODO more stats on N workers (e.g. distance to command centers, distance to minerals, to geysers...)
        return dict(
            num_workers=num_workers,
			num_idle_workers=num_idle_workers,
            pct_idle_workers=pct_idle_workers,
            num_mineral_harvesters=num_mineral_harvesters,
            pct_mineral_harvesters=pct_mineral_harvesters,
        )

    def _get_army_state(self) -> Dict[str, int|float]:
        marines = self._get_units(alliances=PlayerRelative.SELF, unit_types=units.Terran.Marine)
        num_marines = len(marines)
        num_idle_marines = len([m for m in marines if self.is_idle(m)])
        pct_idle_marines = 0 if num_marines == 0 else num_idle_marines / num_marines
        total_army_health = sum(map(lambda b: b.health, marines))

        enemy_army = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.ARMY_UNIT_TYPES)
        enemy_workers = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.WORKER_UNIT_TYPES)
        enemy_buildings = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.BUILDING_UNIT_TYPES)

        marines_mean_position = self._get_mean_position(marines)
        enemy_army_mean_position = self._get_mean_position(enemy_army)
        enemy_workers_mean_position = self._get_mean_position(enemy_workers)
        enemy_buildings_mean_position = self._get_mean_position(enemy_buildings)

        dist_marine_avg_to_army_avg = self._get_distance(marines_mean_position, enemy_army_mean_position)\
            if marines_mean_position is not None and enemy_army_mean_position is not None else -1
        dist_marine_avg_to_worker_avg = self._get_distance(marines_mean_position, enemy_workers_mean_position)\
            if marines_mean_position is not None and enemy_workers_mean_position is not None else -1
        dist_marine_avg_to_building_avg = self._get_distance(marines_mean_position, enemy_buildings_mean_position)\
            if marines_mean_position is not None and enemy_buildings_mean_position is not None else -1

        _, dist_marine_avg_to_closest_army = self._get_closest(enemy_army, marines_mean_position)
        if dist_marine_avg_to_closest_army is None: dist_marine_avg_to_closest_army = -1
        _, dist_marine_avg_to_closest_worker = self._get_closest(enemy_workers, marines_mean_position)
        if dist_marine_avg_to_closest_worker is None: dist_marine_avg_to_closest_worker = -1
        _, dist_marine_avg_to_closest_building = self._get_closest(enemy_buildings, marines_mean_position)
        if dist_marine_avg_to_closest_building is None: dist_marine_avg_to_closest_building = -1

        return dict(
            num_marines=num_marines,
			num_idle_marines=num_idle_marines,
            pct_idle_marines=pct_idle_marines,
            total_army_health=total_army_health,
            dist_marine_avg_to_army_avg=dist_marine_avg_to_army_avg,
            dist_marine_avg_to_worker_avg=dist_marine_avg_to_worker_avg,
            dist_marine_avg_to_building_avg=dist_marine_avg_to_building_avg,
            dist_marine_avg_to_closest_army=dist_marine_avg_to_closest_army,
            dist_marine_avg_to_closest_worker=dist_marine_avg_to_closest_worker,
            dist_marine_avg_to_closest_building=dist_marine_avg_to_closest_building,
        )

    def _get_resources_state(self, obs: TimeStep) -> Dict[str, int|float]:
        return dict(
            free_supply=self.get_free_supply(obs),
            minerals=obs.observation.player.minerals,
            collection_rate_minerals=obs.observation.score_cumulative.collection_rate_minerals/60
        )

    def _get_scores_state(self, obs: TimeStep)     -> Dict[str, int|float]:
        return {
            "score_cumulative_score": obs.observation.score_cumulative.score,
            "score_cumulative_total_value_units": obs.observation.score_cumulative.total_value_units,
            "score_cumulative_total_value_structures": obs.observation.score_cumulative.total_value_structures,
            "score_cumulative_killed_value_units": obs.observation.score_cumulative.killed_value_units,
            "score_cumulative_killed_value_structures": obs.observation.score_cumulative.killed_value_structures,
            # Supply (food) scores
            "score_food_used_none": obs.observation.score_by_category.food_used.none,
            "score_food_used_army": obs.observation.score_by_category.food_used.army,
            "score_food_used_economy": obs.observation.score_by_category.food_used.economy,
            # Used minerals and vespene
            "score_used_minerals_none": obs.observation.score_by_category.used_minerals.none,
            "score_used_minerals_army": obs.observation.score_by_category.used_minerals.army,
            "score_used_minerals_economy": obs.observation.score_by_category.used_minerals.economy,
            "score_used_minerals_technology": obs.observation.score_by_category.used_minerals.technology,
            # Score by vital
            "score_by_vital_total_damage_dealt_life": obs.observation.score_by_vital.total_damage_dealt.life,
            "score_by_vital_total_damage_taken_life": obs.observation.score_by_vital.total_damage_taken.life,
        }

    def _get_neutral_units_state(self) -> Dict[str, int|float]:
        minerals = [unit for unit in self._get_units(alliances=PlayerRelative.NEUTRAL) if Minerals.contains(unit.unit_type)]
        return dict(
            num_minerals=len(minerals),
        )

    def _get_enemy_state(self) -> Dict[str, int|float]:
        enemy_buildings = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.BUILDING_UNIT_TYPES)
        enemy_workers = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.WORKER_UNIT_TYPES)
        enemy_army = self._get_units(alliances=PlayerRelative.ENEMY, unit_types=Constants.ARMY_UNIT_TYPES)

        return dict(
            enemy_total_building_health=sum(map(lambda b: b.health, enemy_buildings)),
            enemy_total_worker_health=sum(map(lambda b: b.health, enemy_workers)),
            enemy_total_army_health=sum(map(lambda b: b.health, enemy_army)),
        )

    def _get_actions_state(self) -> Dict[str, int]:
        return dict(
            can_harvest_minerals=int(AllActions.HARVEST_MINERALS in self._available_actions),
            can_recruit_worker_0=int(AllActions.RECRUIT_SCV_0 in self._available_actions),
            can_recruit_worker_1=int(AllActions.RECRUIT_SCV_1 in self._available_actions),
            can_recruit_worker_2=int(AllActions.RECRUIT_SCV_2 in self._available_actions),
            can_build_supply_depot=int(AllActions.BUILD_SUPPLY_DEPOT in self._available_actions),
            can_build_command_center=int(AllActions.BUILD_COMMAND_CENTER in self._available_actions),
            can_build_barracks=int(AllActions.BUILD_BARRACKS in self._available_actions),
            can_recruit_marine=int(AllActions.RECRUIT_MARINE in self._available_actions),
            can_attack_closest_buildings=int(AllActions.ATTACK_CLOSEST_BUILDING in self._available_actions),
            can_attack_closest_workers=int(AllActions.ATTACK_CLOSEST_WORKER in self._available_actions),
            can_attack_closest_army=int(AllActions.ATTACK_CLOSEST_ARMY in self._available_actions),
            can_attack_buildings=int(AllActions.ATTACK_BUILDINGS in self._available_actions),
            can_attack_workers=int(AllActions.ATTACK_WORKERS in self._available_actions),
            can_attack_army=int(AllActions.ATTACK_ARMY in self._available_actions),
        )
