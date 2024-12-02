from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
from torch import optim


class DQNNetwork(nn.Module, ABC):
    """Base class for Deep Q-Networks with discrete actions.

    This class makes no assumptions over the type of network, and it will
    also not check that the input/output features of the network layers
    matches the shape of the observation /action space.
    """
    def __init__(self, model_layers: List[nn.Module] | List[int], observation_space_shape: int, num_actions: int, learning_rate: float, lr_milestones: list[int] = None, **kwargs):
        super().__init__(**kwargs)

        self.input_shape = observation_space_shape #env.observation_space.shape[0]
        self.n_outputs = num_actions #env.action_space.n
        self.actions = list(range(self.n_outputs))

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch_directml.device():
            self.device = torch_directml.device()
        else:
            self.device = 'cpu'

        if isinstance(model_layers[0], int):
            model_layers = self._get_model_layers_from_number_of_units(layer_units=model_layers, num_inputs=observation_space_shape, num_outputs=num_actions)

        model = torch.nn.Sequential(*model_layers)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if lr_milestones is not None and any(lr_milestones):
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
        else:
            scheduler = None

        self.model = model.to(device=self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler


    def _get_model_layers_from_number_of_units(self, layer_units: List[int], num_inputs: int, num_outputs: int):
        layer_units = layer_units or self.DEFAULT_LAYER_UNITS
        model_layers = []
        in_features = num_inputs

        for output_units in layer_units:
            model_layers.extend(
                [
                    nn.Linear(in_features=in_features, out_features=output_units, bias=True),
                    torch.nn.ReLU()
                ]
            )
            in_features = output_units

        model_layers.append(nn.Linear(in_features=in_features, out_features=num_outputs, bias=True))

        return model_layers

    def get_random_action(self, valid_actions: List = None) -> int:
        """Select a random action.

        Returns:
            int: Selected action
        """
        if valid_actions is not None:
            return np.random.choice(np.where(valid_actions == 1)[0])

        return np.random.choice(self.actions)

    def get_greedy_action(self, state: Union[np.ndarray, list, tuple], valid_actions: List = None):
        state_tensor = torch.Tensor(state).to(device=self.device)
        qvals = self.get_qvals(state_tensor)
        if valid_actions is not None:
            invalid_indices = np.where(valid_actions == 0)[0]
            qvals[invalid_indices] = -torch.inf
        # Action selected = index of the highest Q-value
        action = torch.max(qvals, dim=-1)[1].item()

        return action

    def get_action(self, state: Union[np.ndarray, list, tuple], epsilon: float = 0.05, valid_actions: List = None) -> int:
        """Select an action using epsilon-greedy method.

        Args:
            state (Union[np.ndarray, list, tuple]): Current state.
            epsilon (float, optional): Probability of taking a random action. Defaults to 0.05.

        Returns:
            int: Action selected.
        """
        if np.random.random() < epsilon:
            action = self.get_random_action(valid_actions=valid_actions)
        else:
            action = self.get_greedy_action(state=state, valid_actions=valid_actions)

        return action

    def get_qvals(self, state: torch.Tensor) -> torch.Tensor:
        """Get the Q-values for a certain state.

        Args:
            state (torch.Tensor): State to get the q-values for.

        Returns:
            torch.Tensor: Tensor with the Q-value for each of the actions.
        """
        out = self.model(state)

        return out
