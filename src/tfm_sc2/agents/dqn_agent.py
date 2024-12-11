import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from pysc2.env.environment import TimeStep
from typing_extensions import override, Self

from .stats import EpisodeStats, AgentStats, AggregatedEpisodeStats
from ..actions import AllActions
from ..networks.dqn_network import DQNNetwork
from ..networks.experience_replay_buffer import ExperienceReplayBuffer
from ..types import AgentStage, DQNAgentParams, RewardMode
from .base_agent import BaseAgent


class DQNAgent(BaseAgent):
    _BUFFER_FILE: str = "buffer.pkl"
    _MAIN_NETWORK_FILE: str = "main_network.pt"
    _TARGET_NETWORK_FILE: str = "target_network.pt"

    def __init__(self, main_network: DQNNetwork,
                 hyperparams: DQNAgentParams,
                 target_network: DQNNetwork = None,
                 buffer: ExperienceReplayBuffer = None,
                 random_mode: bool = False,
                 **kwargs):
        """Deep Q-Network agent.

        Args:
            main_network (nn.Module): Main network
            buffer (ExperienceReplayBuffer): Memory buffer
            hyperparams (DQNAgentParams): Agent hyper parameters.
            target_network (nn.Module, optional): Target network. If not provided, then the main network will be cloned.
        """
        super().__init__(**kwargs)

        self.main_network = main_network
        self.target_network = target_network or deepcopy(main_network)
        self._buffer = buffer
        self.hyperparams = hyperparams
        self.initial_epsilon = hyperparams.epsilon
        self.epsilon = hyperparams.epsilon
        self.loss = self.hyperparams.loss or torch.nn.HuberLoss()
        self._random_mode = random_mode
        self._burnin = False

        # Placeholders
        self._action_to_idx = None
        self._idx_to_action = None
        self._num_actions = None

        self.__prev_reward = None

        # Add some extra flags to the default ones (train / exploit)
        self._status_flags.update(dict(
            burnin_started=False,
            main_net_updated=False,
            target_net_updated=False,
        ))

    def burnin(self):
        """Set the agent in burnin mode."""
        self._train = True
        self._burnin = True
        self._exploit = False

    @override
    def train(self):
        """Set the agent in training mode."""
        self._train = True
        self._burnin = False
        self._exploit = False

    @override
    def exploit(self):
        """Set the agent in training mode."""
        self._train = False
        self._burnin = False
        self._exploit = True

    @property
    def _collect_stats(self) -> bool:
        return not (self.is_burnin or self._random_mode)

    @classmethod
    def _extract_init_arguments(cls, checkpoint_path: Path, agent_attrs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        parent_attrs = super()._extract_init_arguments(agent_attrs=agent_attrs, **kwargs)
        main_network_path = checkpoint_path / cls._MAIN_NETWORK_FILE
        target_network_path = checkpoint_path / cls._TARGET_NETWORK_FILE
        return dict(
            **parent_attrs,
            main_network=torch.load(main_network_path),
            target_network=torch.load(target_network_path),
            buffer=agent_attrs["buffer"],
            random_mode=agent_attrs.get("random_mode", agent_attrs.get("random_model", False)),
            hyperparams=agent_attrs["hyperparams"],
        )

    def _load_agent_attrs(self, agent_attrs: Dict):
        super()._load_agent_attrs(agent_attrs)
        self.initial_epsilon = agent_attrs["initial_epsilon"]
        self._num_actions = agent_attrs["num_actions"]
        self.loss = agent_attrs["loss"]
        if "epsilon" in agent_attrs:
            self.epsilon = agent_attrs["epsilon"]

    def _get_agent_attrs(self):
        parent_attrs = super()._get_agent_attrs()
        return dict(
            hyperparams=self.hyperparams,
            buffer=self._buffer,
            initial_epsilon=self.initial_epsilon,
            epsilon=self.epsilon,
            num_actions=self._num_actions,
            loss=self.loss,
            random_mode=self._random_mode,
            **parent_attrs
        )

    def save(self, checkpoint_path: Union[str|Path] = None):
        super().save(checkpoint_path=checkpoint_path)
        main_network_path = checkpoint_path / self._MAIN_NETWORK_FILE
        target_network_path = checkpoint_path / self._TARGET_NETWORK_FILE
        torch.save(self.main_network, main_network_path)
        torch.save(self.target_network, target_network_path)

        buffer_path = checkpoint_path / self._BUFFER_FILE

        if self._buffer is not None:
            with open(buffer_path, "wb") as f:
                pickle.dump(self._buffer, f)
                self.logger.info(f"Saved memory replay buffer to {buffer_path}")

    @classmethod
    def load(cls, checkpoint_path: Union[str|Path], map_name: str, map_config: Dict, buffer: ExperienceReplayBuffer = None, **kwargs) -> Self:
        checkpoint_path = Path(checkpoint_path)
        agent_attrs_file = checkpoint_path / cls._AGENT_FILE
        with open(agent_attrs_file, mode="rb") as f:
            agent_attrs = pickle.load(f)

        if buffer is not None:
            agent_attrs["buffer"] = buffer

        init_attrs = cls._extract_init_arguments(checkpoint_path=checkpoint_path, agent_attrs=agent_attrs, map_name=map_name, map_config=map_config)
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

    def _current_agent_stage(self):
        if self.is_burnin:
            return AgentStage.BURN_IN
        if self._exploit:
            return AgentStage.EXPLOIT
        if self.is_training:
            return AgentStage.TRAINING
        return AgentStage.UNKNOWN

    @property
    def burn_in_capacity(self) -> float:
        return self._buffer.burn_in_capacity

    @property
    def memory_replay_ready(self) -> bool:
        return self._buffer.burn_in_capacity >= 1

    def reset(self, **kwargs):
        """Initialize the agent."""
        super().reset(**kwargs)
        # Last observation
        self.__prev_reward = None

        self._current_episode_stats.is_burnin = self.is_burnin

    @property
    def is_training(self):
        return super().is_training and (not self._random_mode) and (not self.is_burnin)

    @property
    def is_burnin(self):
        return self._burnin

    def select_action(self, obs: TimeStep) -> Tuple[AllActions, Dict[str, Any], bool]:
        if self._action_masking:
            available_actions = self._available_actions
        else:
            available_actions = [a for a in self.agent_actions if a in self._map_config["available_actions"]]

        # One-hot encoded version of available actions
        valid_actions = self._actions_to_network(available_actions)
        if not any(valid_actions):
            valid_actions = None
        if (not self._exploit) and (self.is_burnin or self._random_mode):
            if self._random_mode:
                self.logger.debug(f"Random mode - collecting experience from random actions")
            elif not self._status_flags["burnin_started"]:
                self.logger.info(f"Starting burn-in")
                self._status_flags["burnin_started"] = True
            else:
                self.logger.debug(f"Burn in capacity: {100 * self._buffer.burn_in_capacity:.2f}%")

            raw_action = self.main_network.get_random_action(valid_actions=valid_actions)
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

        action_args, is_valid_action = self._get_action_args(obs=obs, action=action)

        return action, action_args, is_valid_action

    def pre_step(self, obs: TimeStep, eval_step: bool = True):
        super().pre_step(obs, eval_step)

        done = obs.last()
        if eval_step:
            if not self._exploit and self._buffer is not None:
                ohe_available_actions = self._actions_to_network(self._available_actions)
                self._buffer.append(self._prev_state_tuple, self._prev_action, self._prev_action_args,
                                    self._current_reward, self._current_adjusted_reward, self._current_score, done,
                                    self._current_state_tuple, ohe_available_actions)

            # do updates
            if (not self.is_burnin) and self.is_training:
                main_net_updated = False
                target_net_updated = False
                if self.hyperparams.main_network_update_frequency > 0:
                    if (self.current_agent_stats.step_count % self.hyperparams.main_network_update_frequency) == 0:
                        if not self._status_flags["main_net_updated"]:
                            self.logger.debug(f"First main network update with lr {self.main_network.learning_rate}")
                            self._status_flags["main_net_updated"] = True
                        else:
                            self.logger.debug(f"Main network update with lr {self.main_network.learning_rate}")
                        self._current_episode_stats.losses = self.update_main_network(self._current_episode_stats.losses)
                        main_net_updated = True
                if self.hyperparams.target_network_sync_frequency > 0:
                    if (self.current_agent_stats.step_count % self.hyperparams.target_network_sync_frequency) == 0:
                        if not self._status_flags["target_net_updated"]:
                            self.logger.debug(f"First target network update with lr {self.target_network.learning_rate}")
                            self._status_flags["target_net_updated"] = True
                        else:
                            self.logger.debug(f"Target network update with lr {self.target_network.learning_rate}")
                        self.synchronize_target_network()
                        target_net_updated = True
                # HERE

                if done:
                    if not main_net_updated:
                        # If we finished but didn't update, perform one last update
                        self._current_episode_stats.losses = self.update_main_network(self._current_episode_stats.losses)
                    if not target_net_updated:
                        self.synchronize_target_network()

        if done:
            if self.is_training:
                self._current_episode_stats.epsilon = self.epsilon if not self._random_mode else 1.
                self._current_episode_stats.loss = np.mean(self._current_episode_stats.losses)
                self.epsilon = max(self.epsilon * self.hyperparams.epsilon_decay, self.hyperparams.min_epsilon)
                self.main_network.step_lr()
                self.target_network.step_lr()
            elif self._exploit:
                self._current_episode_stats.epsilon = 0.
            else:
                self._current_episode_stats.epsilon = 1.

    def post_step(self, obs: TimeStep, action: AllActions, action_args: Dict[str, Any], original_action: AllActions, original_action_args: Dict[str, Any], is_valid_action: bool):
        if obs.last():
            self._current_episode_stats.is_random_mode = self._random_mode

        super().post_step(obs, action, action_args, original_action, original_action_args, is_valid_action)

    def _get_end_of_episode_info_components(self) -> List[str]:
        return super()._get_end_of_episode_info_components() + [
            f"Epsilon {self._current_episode_stats.epsilon:.4f}",
            f"Mean Loss ({len(self._current_episode_stats.losses)} updates) {self._current_episode_stats.mean_loss:.4f}",
        ]

    def _calculate_loss(self, batch: Iterable[Tuple]) -> torch.Tensor:
        """Calculate the loss for a batch.

        Args:
            batch (Iterable[Tuple]): Batch to calculate the loss on.

        Returns:
            torch.Tensor: The calculated loss between the calculated and the predicted values.
        """

        # Convert elements from the replay buffer to tensors
        states, actions, action_args, rewards, adjusted_rewards, scores, dones, next_states, next_state_available_actions = [i for i in batch]
        states = torch.stack([torch.Tensor(state) for state in states]).to(device=self.device)
        next_states = torch.stack([torch.Tensor(state) for state in next_states]).to(device=self.device)
        match self._reward_mode:
            case RewardMode.SCORE:
                rewards_to_use = scores
            case RewardMode.ADJUSTED_REWARD:
                rewards_to_use = adjusted_rewards
            case RewardMode.REWARD:
                rewards_to_use = rewards
            case _:
                self.logger.warning(f"Unknown reward method {self._reward_mode.name} - using default rewards")
                rewards_to_use = rewards
        rewards_vals = torch.FloatTensor(rewards_to_use).to(device=self.device)
        actions_vals = torch.LongTensor(np.array(actions)).to(device=self.device).reshape(-1,1)
        dones_t = torch.BoolTensor(dones).to(device=self.device)

        # Get q-values from the main network
        qvals = torch.gather(self.main_network.get_qvals(states), 1, actions_vals)
        # Get q-values from the target network
        if self._action_masking:
            # Select max q-value from available actions for each experience
            nsaa_mask = torch.BoolTensor(np.array(next_state_available_actions)).to(device=self.device)
            masked_qvals = torch.where(nsaa_mask, self.target_network.get_qvals(next_states), -torch.inf)
            qvals_next = torch.max(masked_qvals, dim=-1)[0].detach()
        else:
            # Select max q-value for each experience
            qvals_next = torch.max(self.target_network.get_qvals(next_states), dim=-1)[0].detach()
        # Set terminal states to 0
        qvals_next[dones_t] = 0

        # Get expected q-values
        expected_qvals = self.hyperparams.gamma * qvals_next + rewards_vals

        return self.loss(qvals, expected_qvals.reshape(-1,1))

    def update_main_network(self, episode_losses: List[float] = None) -> List[float]:
        """Update the main network.

        Normally we only perform the update every certain number of steps, defined by
        main_network_update_frequency, but if force_update is set to true, then
        the update will happen, independently of the current step count.

        Args:
            episode_losses (List, optional): List with episode losses.
        Returns:
            List[float]: A list with all the losses in the current episode.
        """
        episode_losses = episode_losses or []

        self.main_network.optimizer.zero_grad()  # Remove previous gradients
        batch = self._buffer.sample_batch(batch_size=self.hyperparams.batch_size) # Sample experiences from the buffer

        loss = self._calculate_loss(batch)# Get batch loss
        loss.backward() # Backward pass to get gradients

        self.main_network.step()

        if self.device != 'cpu':
            loss = loss.detach().cpu().numpy()
        else:
            loss = loss.detach().numpy()

        episode_losses.append(float(loss))
        return episode_losses

    def synchronize_target_network(self):
        """Synchronize the target network with the main network parameters.

        When the target_sync_mode is set to "soft", a soft update is made, so instead of overwriting
        the target fully, we update it by mixing in the current target parameters and the main network
        parameters. In practice, we keep a fraction (1 - update_tau) from the target network, and add
        to it a fraction update_tau from the main network.
        """

        if self.hyperparams.target_sync_mode == "soft":
            for target_var, var in zip(self.target_network.parameters(), self.main_network.parameters()):
                    target_var.data.copy_((1. - self.hyperparams.update_tau) * target_var.data + (self.hyperparams.update_tau) * var.data)
        else:
            self.target_network.load_state_dict(self.main_network.state_dict())
