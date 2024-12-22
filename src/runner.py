import configparser
import io
import logging
import time
import pickle

import numpy as np
from absl import app,flags
from pathlib import Path

import torch
from pysc2.env import sc2_env
from pysc2.lib.remote_controller import ConnectError

from tfm_sc2.actions import GameManagerActions, BaseManagerActions, ArmyRecruitManagerActions, ArmyAttackManagerActions, \
    AllActions
from tfm_sc2.agents.multi.army_attack_manager_dqn_agent import ArmyAttackManagerDQNAgent
from tfm_sc2.agents.multi.army_attack_manager_random_agent import ArmyAttackManagerRandomAgent
from tfm_sc2.agents.multi.army_recruit_manager_dqn_agent import ArmyRecruitManagerDQNAgent
from tfm_sc2.agents.multi.army_recruit_manager_random_agent import ArmyRecruitManagerRandomAgent
from tfm_sc2.agents.multi.base_manager_dqn_agent import BaseManagerDQNAgent
from tfm_sc2.agents.multi.base_manager_random_agent import BaseManagerRandomAgent
from tfm_sc2.agents.multi.game_manager_dqn_agent import GameManagerDQNAgent
from tfm_sc2.agents.multi.game_manager_random_agent import GameManagerRandomAgent
from tfm_sc2.agents.single.single_dqn_agent import SingleDQNAgent
from tfm_sc2.agents.single.single_random_agent import SingleRandomAgent
from tfm_sc2.agents.single.single_scripted_agent import SingleScriptedAgent
from tfm_sc2.networks.dqn_network import DQNNetwork
from tfm_sc2.networks.experience_replay_buffer import ExperienceReplayBuffer
from tfm_sc2.sc2_config import MAP_CONFIGS, SC2_CONFIG

from codecarbon import OfflineEmissionsTracker

from tfm_sc2.types import RewardMode, State, DQNAgentParams
from tfm_sc2.with_logger import WithLogger


_CONFIG = configparser.ConfigParser(allow_no_value=True)


class MainLogger:
    _logger = None
    @classmethod
    def get(cls) -> logging.Logger:
        if cls._logger is None:
            cls._logger = logging.getLogger("main_runner")
            # cls._logger.setLevel(logging.INFO)
        return cls._logger


def _setup_logging(log_file: str = None):
    if log_file is not None:
        log_file: Path = Path(log_file)
        log_file.parent.mkdir(exist_ok=True, parents=True)
    WithLogger.init_logging(stream_level=logging.INFO, file_name=log_file, file_level=logging.DEBUG)


def _run_burnin(main_agent, other_agents):
    map_name = _CONFIG.get("train", "map")
    map_config = MAP_CONFIGS[map_name]
    save_path = Path(_CONFIG.get("train", "save_path"))
    export_stats_only = _CONFIG.getboolean("train", "export_stats_only")
    max_burnin_episodes = _CONFIG.getint("train", "max_burnin_episodes")

    logger = MainLogger.get()

    burnin_episodes = 0
    max_episode_failures = 5
    current_episode_failures = 0
    is_burnin = True
    while current_episode_failures < max_episode_failures:
        try:
            with sc2_env.SC2Env(
                    map_name=map_name,
                    players=map_config["players"],
                    **SC2_CONFIG) as env:

                for a in other_agents:
                    a.setup(env.observation_spec(), env.action_spec())
                main_agent.setup(env.observation_spec(), env.action_spec())

                while is_burnin and (burnin_episodes < max_burnin_episodes):
                    timesteps = env.reset()
                    for a in other_agents:
                        a.reset()
                    main_agent.reset()
                    episode_ended = timesteps[0].last()
                    while not episode_ended:
                        step_actions = [a.step(timestep) for a, timestep in zip([main_agent, *other_agents], timesteps)]
                        timesteps = env.step(step_actions)
                        episode_ended = timesteps[0].last()

                        if episode_ended:
                            # Perform one last step to process rewards etc
                            last_step_actions = [a.step(timestep) for a, timestep in zip([main_agent, *other_agents], timesteps)]

                    burnin_episodes += 1
                    current_episode_failures = 0
                    is_burnin = not main_agent.memory_replay_ready

                    logger.info(f"Burnin progress: {100 * main_agent.burn_in_capacity:.2f}%")
            break

        except ConnectError as error:
            logger.warning("Couldn't connect to SC2 environment, trying to restart the episode again")
            logger.warning(error)
            current_episode_failures += 1
            if current_episode_failures >= max_episode_failures:
                logger.error(f"Reached max number of allowed episode failures, stopping run")

    logger.info(f"Finished burnin after {burnin_episodes} episodes")

    if export_stats_only:
        main_agent.save_stats(save_path)
    else:
        main_agent.save(save_path)

    return burnin_episodes


def _train(main_agent, other_agents, tracker: OfflineEmissionsTracker):
    try:
        logger = MainLogger.get()

        # We set measure_power_secs to a very high value because we want to flush emissions as we want
        main_agent.set_tracker(tracker)
        tracker.start()

        if hasattr(main_agent, "memory_replay_ready"):
            perform_burnin = not main_agent.memory_replay_ready
            logger.info(f"Agent has a memory replay buffer. Requires burn-in: {perform_burnin}")
            if perform_burnin:
                main_agent.burnin()
                _run_burnin(main_agent, other_agents)

        fine_tune = _CONFIG.getboolean("train", "fine_tune")
        if fine_tune:
            main_agent.fine_tune()
        else:
            main_agent.train()
        finished_episodes = _run_episodes(main_agent, other_agents, "train")
        total_emissions = tracker.stop()
        logger.info(f"Total emissions after {finished_episodes} episodes for agent {main_agent._log_name} (and {len(other_agents)} other agents): {total_emissions:.2f}")

        num_episodes = _CONFIG.getint("train", "num_episodes")
        return finished_episodes == num_episodes

    except KeyboardInterrupt:
        pass


def _run_episodes(main_agent, other_agents, mode):
    map_name = _CONFIG.get(mode, "map")
    map_config = MAP_CONFIGS[map_name]
    save_path = Path(_CONFIG.get(mode, "save_path"))
    export_stats_only = _CONFIG.getboolean(mode, "export_stats_only")
    num_episodes = _CONFIG.getint(mode, "num_episodes")
    save_frequency_episodes = _CONFIG.getint(mode, "save_frequency_episodes")
    early_stopping_interval = _CONFIG.getint(mode, "early_stopping_interval")
    reward_mode = _reward_mode_to_enum(_CONFIG.get(mode, "reward_mode"))

    logger = MainLogger.get()

    finished_episodes = 0
    max_episode_failures = 5
    current_episode_failures = 0
    num_wins, num_draws, num_losses = 0, 0, 0
    prev_mean_reward = -np.inf
    episodes_without_improvement = 0
    logger.info(f"Beginning agent {mode}")
    while current_episode_failures < max_episode_failures:
        try:
            with sc2_env.SC2Env(
                    map_name=map_name,
                    players=map_config["players"],
                    **SC2_CONFIG) as env:

                for a in other_agents:
                    a.setup(env.observation_spec(), env.action_spec())
                main_agent.setup(env.observation_spec(), env.action_spec())

                while finished_episodes < num_episodes:
                    logger.info(f"Starting episode {finished_episodes + 1}")
                    t0 = time.time()
                    already_saved = False

                    timesteps = env.reset()
                    for a in other_agents:
                        a.reset()
                    main_agent.reset()
                    episode_ended = timesteps[0].last()
                    total_time_actions, total_time_steps, n_steps = 0, 0, 0
                    while not episode_ended:
                        step_t0 = time.time()

                        step_actions = [a.step(timestep) for a, timestep in zip([main_agent, *other_agents], timesteps)]
                        step_t1 = time.time()

                        timesteps = env.step(step_actions)
                        step_t2 = time.time()

                        total_time_actions += step_t1 - step_t0
                        total_time_steps += step_t2 - step_t1
                        n_steps += 1
                        logger.debug(f"Action calculated in {(step_t1 - step_t0) * 1000:-2f}ms")
                        logger.debug(f"Step performed in {(step_t2 - step_t1) * 1000:-2f}ms")

                        episode_ended = timesteps[0].last()

                    # Perform one last step to process rewards etc
                    last_step_actions = [a.step(timestep) for a, timestep in zip([main_agent, *other_agents], timesteps)]

                    finished_episodes += 1
                    current_episode_failures = 0

                    t1 = time.time()
                    t_delta = t1 - t0
                    logger.info(
                        f"Episode {finished_episodes}/{num_episodes} completed in {t_delta:.2f} seconds ({t_delta / 60:.2f} minutes)")
                    logger.info(
                        f"Total time calculating actions: {total_time_actions:.2f}s ({(total_time_actions / n_steps) * 1000} ms/step)")
                    logger.info(
                        f"Total time performing steps: {total_time_steps:.2f}s ({(total_time_steps / n_steps) * 1000} ms/step)")

                    if other_agents:
                        if timesteps[0].reward > 0:
                            num_wins += 1
                        elif timesteps[0].reward < 0:
                            num_losses += 1
                        else:
                            num_draws += 1
                        logger.info(f"Main agent results: [{num_wins}/{num_draws}/{num_losses}]")
                        win_rate = 100 * num_wins / finished_episodes
                        logger.info(f"Win rate for main agent: {win_rate:.2f}%")

                    if finished_episodes % save_frequency_episodes == 0:
                        logger.info(f"Saving agent after {finished_episodes} episodes")
                        if export_stats_only:
                            main_agent.save_stats(save_path)
                        else:
                            main_agent.save(save_path)
                        already_saved = True

                    mean_reward = main_agent.current_aggregated_episode_stats.mean_rewards(
                        stage=main_agent._current_agent_stage().name, last_n=10,
                        reward_mode=reward_mode)
                    if mean_reward <= prev_mean_reward:
                        episodes_without_improvement += 1
                        if 0 < early_stopping_interval <= episodes_without_improvement:
                            logger.info(f"Stopping early after {finished_episodes} episodes")
                            break
                    else:
                        episodes_without_improvement = 0
                    prev_mean_reward = mean_reward
            break

        except ConnectError as error:
            logger.error("Couldn't connect to SC2 environment, trying to restart the episode again")
            logger.error(error)
            current_episode_failures += 1
            if current_episode_failures >= max_episode_failures:
                logger.error(f"Reached max number of allowed episode failures, stopping run")

    logger.info(f"Finished {mode} after {finished_episodes} episodes")

    if not already_saved:
        logger.info(f"Saving final agent after {finished_episodes} episodes")
        if export_stats_only:
            main_agent.save_stats(save_path)
        else:
            main_agent.save(save_path)

    return finished_episodes


def _exploit(main_agent, other_agents):
    try:
        logger = MainLogger.get()

        main_agent.set_tracker(None)
        main_agent.exploit()
        finished_episodes = _run_episodes(main_agent, other_agents, "exploit")

        num_episodes = _CONFIG.getint("exploit", "num_episodes")
        return finished_episodes == num_episodes

    except KeyboardInterrupt:
        pass


def _reward_mode_to_enum(reward_mode_str):
    logger = MainLogger.get()
    reward_mode = RewardMode.from_name(reward_mode_str)
    if reward_mode is None:
        logger.warning(f"Unknown reward_mode flag {reward_mode_str}, falling back to REWARD")
        reward_mode = RewardMode.REWARD
    return reward_mode


def _get_dqn_buffer(config_section):
    logger = MainLogger.get()

    model_path = Path(_CONFIG.get(config_section, "model_path"))

    buffer_file = _CONFIG.get(config_section, "buffer_file")

    if buffer_file is not None:
        buffer_path = model_path / buffer_file
        if buffer_path.exists() and buffer_path.is_file():
            logger.info(f"Using buffer from file {buffer_path}")
            with open(buffer_path, mode="rb") as f:
                buffer = pickle.load(f)

            return buffer

    memory_size = _CONFIG.getint(config_section, "memory_size")
    burn_in = _CONFIG.getint(config_section, "burn_in")

    logger.info(f"Creating new buffer with memory size = {memory_size} and burn-in = {burn_in}")
    buffer = ExperienceReplayBuffer(memory_size=memory_size, burn_in=burn_in)

    return buffer


def _create_random_agent(cls, config_section, log_name):
    map_name = _CONFIG.get(config_section, "map")
    map_config = MAP_CONFIGS[map_name]
    reward_mode = _reward_mode_to_enum(_CONFIG.get(config_section, "reward_mode"))
    score_method = _CONFIG.get(config_section, "score_method")
    action_masking = _CONFIG.getboolean(config_section, "action_masking")

    return cls(map_name=map_name,
               map_config=map_config,
               reward_mode=reward_mode,
               score_method=score_method,
               action_masking=action_masking,
               log_name=log_name,)


def _create_single_agent(cls, config_section, log_name):
    map_name = _CONFIG.get(config_section, "map")
    map_config = MAP_CONFIGS[map_name]
    reward_mode = _reward_mode_to_enum(_CONFIG.get(config_section, "reward_mode"))
    score_method = _CONFIG.get(config_section, "score_method")
    action_masking = _CONFIG.getboolean(config_section, "action_masking")

    return cls(map_name=map_name,
               map_config=map_config,
               reward_mode=reward_mode,
               score_method=score_method,
               action_masking=action_masking,
               log_name=log_name,)


def _create_dqn(cls = None, config_section = None, log_name: str = None):
    logger = MainLogger.get()
    if cls in [GameManagerDQNAgent, GameManagerRandomAgent]:
        num_actions = len(GameManagerActions)
        logger.info(f"Creating DQN with {num_actions} actions from GameManagerActions for agent {cls}")
    elif cls in [BaseManagerDQNAgent, BaseManagerRandomAgent]:
        num_actions = len(BaseManagerActions)
        logger.info(f"Creating DQN with {num_actions} actions from BaseManagerActions for agent {cls}")
    elif cls in [ArmyRecruitManagerDQNAgent, ArmyRecruitManagerRandomAgent]:
        num_actions = len(ArmyRecruitManagerActions)
        logger.info(f"Creating DQN with {num_actions} actions from ArmyRecruitManagerActions for agent {cls}")
    elif cls in [ArmyAttackManagerDQNAgent, ArmyAttackManagerRandomAgent]:
        num_actions = len(ArmyAttackManagerActions)
        logger.info(f"Creating DQN with {num_actions} actions from ArmyAttackManagerActions for agent {cls}")
    else:
        num_actions = len(AllActions)
        logger.info(f"Creating DQN with {num_actions} actions from AllActions for agent {cls}")

    dqn_size = _CONFIG.get(config_section, "dqn_size")
    match dqn_size:
        case "extra_small":
            model_layers = [64, 32]
            logger.info(f"Using extra small network ({model_layers}) for agent {cls}")
        case "small":
            model_layers = [128, 64, 32]
            logger.info(f"Using small network ({model_layers}) for agent {cls}")
        case "medium":
            model_layers = [128, 128, 64, 32]
            logger.info(f"Using medium network ({model_layers}) for agent {cls}")
        case "large":
            model_layers = [256, 128, 128, 64, 32]
            logger.info(f"Using large network ({model_layers}) for agent {cls}")
        case "extra_large":
            model_layers = [512, 256, 128, 128, 64, 32]
            logger.info(f"Using extra large network ({model_layers}) for agent {cls}")

    obs_input_shape = len(State._fields)
    learning_rate = _CONFIG.getfloat(config_section, "lr")
    str_lr_milestones = _CONFIG.get(config_section, "lr_milestones").split(",")
    lr_milestones = [int(n) for n in str_lr_milestones] if any(str_lr_milestones) else []
    dqn = DQNNetwork(model_layers=model_layers,
                     observation_space_shape=obs_input_shape,
                     num_actions=num_actions,
                     learning_rate=learning_rate,
                     lr_milestones=lr_milestones,
                     log_name=log_name)

    return dqn


def _load_dqn(network_path, config_section):
    logger = MainLogger.get()

    dqn = torch.load(network_path)

    if _CONFIG.has_option(config_section, "lr"):
        logger.info("Updating network optimizer")
        learning_rate = _CONFIG.getfloat(config_section, "lr")

        if _CONFIG.has_option(config_section, "lr"):
            str_lr_milestones = _CONFIG.get(config_section, "lr_milestones").split(",")
            lr_milestones = [int(n) for n in str_lr_milestones] if any(str_lr_milestones) else []
        else:
            lr_milestones = []

        dqn.add_optimizer(learning_rate, lr_milestones)

    return dqn


def _get_dqn_params(config_section):
    return DQNAgentParams(epsilon=_CONFIG.getfloat(config_section, "epsilon"),
                          epsilon_decay=_CONFIG.getfloat(config_section, "epsilon_decay"),
                          min_epsilon=_CONFIG.getfloat(config_section, "min_epsilon"),
                          batch_size=_CONFIG.getint(config_section, "batch_size"),
                          gamma=_CONFIG.getfloat(config_section, "gamma"),
                          main_network_update_frequency=_CONFIG.getint(config_section, "main_network_update_frequency"),
                          target_network_sync_frequency=_CONFIG.getint(config_section, "target_network_sync_frequency"),
                          target_sync_mode=_CONFIG.get(config_section, "target_sync_mode"),
                          update_tau=_CONFIG.getfloat(config_section, "update_tau"))


def _create_single_dqn_agent(cls, main_network, target_network, config_section, log_name):
    buffer = _get_dqn_buffer(config_section)
    agent_params = _get_dqn_params(config_section)

    map_name = _CONFIG.get(config_section, "map")
    map_config = MAP_CONFIGS[map_name]
    reward_mode = _reward_mode_to_enum(_CONFIG.get(config_section, "reward_mode"))
    score_method = _CONFIG.get(config_section, "score_method")
    action_masking = _CONFIG.getboolean(config_section, "action_masking")

    return cls(map_name=map_name,
               map_config=map_config,
               main_network=main_network,
               target_network=target_network,
               buffer=buffer,
               hyperparams=agent_params,
               reward_mode=reward_mode,
               action_masking=action_masking,
               score_method=score_method,
               log_name=log_name,)


def _load_single_dqn_agent(cls, config_section, log_name):
    logger = MainLogger.get()
    model_path = Path(_CONFIG.get(config_section, "model_path"))

    load_networks_only = _CONFIG.getboolean(config_section, "load_networks_only")
    if load_networks_only:
        main_network_file = Path(_CONFIG.get(config_section, "main_network_file"))
        target_network_file = Path(_CONFIG.get(config_section, "target_network_file"))
        logger.info(f"Loading main network from file {main_network_file}")
        main_network = _load_dqn(main_network_file, config_section)
        logger.info(f"Loading target network from file {target_network_file}")
        target_network = _load_dqn(target_network_file, config_section)
        return _create_single_dqn_agent(cls, main_network, target_network, config_section, log_name)
    else:
        logger.info(f"Loading agent from file {model_path / cls._AGENT_FILE}")

        map_name = _CONFIG.get(config_section, "map")
        map_config = MAP_CONFIGS[map_name]

        buffer = _get_dqn_buffer(config_section)

        return cls.load(checkpoint_path=model_path,
                        map_name=map_name,
                        map_config=map_config,
                        buffer=buffer)


def _get_single_dqn_agent(cls, config_section, log_name):
    logger = MainLogger.get()
    load_agent = _CONFIG.getboolean(config_section, "load_agent")

    if load_agent:
        agent_file = Path(_CONFIG.get(config_section, "agent_file"))
        logger.info(f"Agent will be loaded from file: {agent_file}")
        return _load_single_dqn_agent(cls, config_section, log_name)
    else:
        logger.info(f"A new agent will be created")
        main_network = _create_dqn(cls, config_section, log_name + " - DQNNetwork")
        return _create_single_dqn_agent(cls, main_network, None, config_section, log_name)


def _create_random_subagents(config_section, log_name):
    base_manager_subagent = _create_random_agent(BaseManagerRandomAgent,
                                                 config_section + ".subagent.base_manager",
                                                 "Sub Agent - Random BaseManager")
    army_recruit_manager_subagent = _create_random_agent(ArmyRecruitManagerRandomAgent,
                                                         config_section + ".subagent.army_recruit_manager",
                                                         "Sub Agent - Random ArmyRecruitManager")
    army_attack_manager_subagent = _create_random_agent(ArmyAttackManagerRandomAgent,
                                                        config_section + ".subagent.army_attack_manager",
                                                        "Sub Agent - Random ArmyAttackManager")

    return base_manager_subagent, army_recruit_manager_subagent, army_attack_manager_subagent


def _get_subagents(config_section, log_name):
    base_manager_subagent = _get_single_dqn_agent(BaseManagerDQNAgent,
                                                  config_section + ".subagent.base_manager",
                                                  "Sub Agent - BaseManager")
    army_recruit_manager_subagent = _get_single_dqn_agent(ArmyRecruitManagerDQNAgent,
                                                          config_section + ".subagent.army_recruit_manager",
                                                          "Sub Agent - ArmyRecruitManager")
    army_attack_manager_subagent = _get_single_dqn_agent(ArmyAttackManagerDQNAgent,
                                                         config_section + ".subagent.army_attack_manager",
                                                         "Sub Agent - ArmyAttackManager")

    return base_manager_subagent, army_recruit_manager_subagent, army_attack_manager_subagent


def _create_multi_dqn_agent(cls, base_manager, army_recruit_manager, army_attack_manager, main_network, target_network, config_section, log_name):
    buffer = _get_dqn_buffer(config_section)
    agent_params = _get_dqn_params(config_section)

    map_name = _CONFIG.get(config_section, "map")
    map_config = MAP_CONFIGS[map_name]
    reward_mode = _reward_mode_to_enum(_CONFIG.get(config_section, "reward_mode"))
    score_method = _CONFIG.get(config_section, "score_method")
    action_masking = _CONFIG.getboolean(config_section, "action_masking")
    time_displacement = _CONFIG.getint(config_section, "time_displacement")

    return cls(map_name=map_name,
               map_config=map_config,
               base_manager=base_manager,
               army_recruit_manager=army_recruit_manager,
               army_attack_manager=army_attack_manager,
               time_displacement=time_displacement,
               main_network=main_network,
               target_network=target_network,
               buffer=buffer,
               hyperparams=agent_params,
               reward_mode=reward_mode,
               action_masking=action_masking,
               score_method=score_method,
               log_name=log_name,)


def _load_multi_dqn_agent(cls, base_manager, army_recruit_manager, army_attack_manager, config_section, log_name):
    logger = MainLogger.get()
    model_path = Path(_CONFIG.get(config_section, "model_path"))

    load_networks_only = _CONFIG.getboolean(config_section, "load_networks_only")
    if load_networks_only:
        main_network_file = Path(_CONFIG.get(config_section, "main_network_file"))
        target_network_file = Path(_CONFIG.get(config_section, "target_network_file"))
        logger.info(f"Loading main network from file {main_network_file}")
        main_network = _load_dqn(main_network_file, config_section)
        logger.info(f"Loading target network from file {target_network_file}")
        target_network = _load_dqn(target_network_file, config_section)
        return _create_multi_dqn_agent(cls,
                                       base_manager,
                                       army_recruit_manager,
                                       army_attack_manager,
                                       main_network, target_network,
                                       config_section, log_name)
    else:
        logger.info(f"Loading agent from file {model_path / cls._AGENT_FILE}")

        map_name = _CONFIG.get(config_section, "map")
        map_config = MAP_CONFIGS[map_name]

        buffer = _get_dqn_buffer(config_section)

        return cls.load(checkpoint_path=model_path,
                        map_name=map_name,
                        map_config=map_config,
                        base_manager=base_manager,
                        army_recruit_manager=army_recruit_manager,
                        army_attack_manager=army_attack_manager,
                        buffer=buffer)


def _get_multi_dqn_agent(cls, config_section, log_name):
    logger = MainLogger.get()
    load_agent = _CONFIG.getboolean(config_section, "load_agent")

    use_random_subagents = _CONFIG.getboolean(config_section, "use_random_subagents")
    if use_random_subagents:
        (base_manager,
         army_recruit_manager,
         army_attack_manager) = _create_random_subagents(config_section, log_name)
    else:
        (base_manager,
         army_recruit_manager,
         army_attack_manager) = _get_subagents(config_section, log_name)

    # base_manager.logger.setLevel(logging.WARNING)
    # army_recruit_manager.logger.setLevel(logging.WARNING)
    # army_attack_manager.logger.setLevel(logging.WARNING)

    if load_agent:
        agent_file = Path(_CONFIG.get(config_section, "agent_file"))
        logger.info(f"Agent will be loaded from file: {agent_file}")
        return _load_multi_dqn_agent(cls,
                                     base_manager,
                                     army_recruit_manager,
                                     army_attack_manager,
                                     config_section, log_name)
    else:
        logger.info(f"A new agent will be created")

        main_network = _create_dqn(cls, config_section, log_name + " - DQNNetwork")
        return _create_multi_dqn_agent(cls,
                                       base_manager,
                                       army_recruit_manager,
                                       army_attack_manager,
                                       main_network, None,
                                       config_section, log_name)


def _get_agent(config_section):
    logger = MainLogger.get()

    agent_key = _CONFIG.get(config_section, "agent_key")

    match agent_key:
        case "single.random":
            log_name = "Main Agent - Random"
            agent = _create_random_agent(SingleRandomAgent, config_section, log_name)
            return agent
        case "single.scripted":
            log_name = "Main Agent - Scripted"
            # agent = get_scripted_agent(SingleScriptedAgent, config_section, log_name)
            # return agent
        case "single.dqn":
            log_name = "Main Agent - SingleDQN"
            agent = _get_single_dqn_agent(SingleDQNAgent, config_section, log_name)
            return agent
        case "multi.random.base_manager":
            log_name = "Main Agent - Random BaseManager"
            agent = _create_random_agent(BaseManagerRandomAgent, config_section, log_name)
            return agent
        case "multi.random.army_recruit_manager":
            log_name = "Main Agent - Random ArmyRecruitManager"
            agent = _create_random_agent(ArmyRecruitManagerRandomAgent, config_section, log_name)
            return agent
        case "multi.random.army_attack_manager":
            log_name = "Main Agent - Random ArmyAttackManager"
            agent = _create_random_agent(ArmyAttackManagerRandomAgent, config_section, log_name)
            return agent
        case "multi.random.game_manager":
            # use_random_subagents = _CONFIG.getboolean(config_section, "use_random_subagents")
            # if use_random_subagents:
            #     log_name = "Main Agent - Random BaseManager"
            #     _create_random_agent(BaseManagerRandomAgent, config_section, log_name)
            #
            #     log_name = "Main Agent - Random ArmyRecruitManager"
            #     _create_random_agent(ArmyRecruitManagerRandomAgent, config_section, log_name)
            #
            #     log_name = "Main Agent - Random ArmyAttackManager"
            #     _create_random_agent(ArmyAttackManagerRandomAgent, config_section, log_name)
            #     pass
            #
            log_name = "Main Agent - Random GameManager"
        case "multi.dqn.base_manager":
            log_name = "Main Agent - Base Manager"
            agent = _get_single_dqn_agent(BaseManagerDQNAgent, config_section, log_name)
            return agent
        case "multi.dqn.army_recruit_manager":
            log_name = "Main Agent - Army Manager"
            agent = _get_single_dqn_agent(ArmyRecruitManagerDQNAgent, config_section, log_name)
            return agent
        case "multi.dqn.army_attack_manager":
            log_name = "Main Agent - Attack Manager"
            agent = _get_single_dqn_agent(ArmyAttackManagerDQNAgent, config_section, log_name)
            return agent
        case "multi.dqn.game_manager":
            log_name = "Main Agent - GameManager"
            agent = _get_multi_dqn_agent(GameManagerDQNAgent, config_section, log_name)
            return agent
        case _:
            raise RuntimeError(f"Unknown agent key {agent_key}")


def _get_tracker(config_section):
    save_path = Path(_CONFIG.get(config_section, "save_path"))
    model_id = _CONFIG.get(config_section, "model_id")
    map_name = _CONFIG.get(config_section, "map")

    emissions_filename = "emissions.csv"
    emissions_file = save_path / emissions_filename
    emissions_idx = 0
    while emissions_file.exists():
        emissions_idx += 1
        if emissions_idx >= 100:
            raise RuntimeError(
                f"There are already 100 emission files under {save_path}, please clean them up or move them to continue")
        emissions_filename = f"emissions_{emissions_idx:02d}.csv"
        emissions_file = save_path / emissions_filename
    return OfflineEmissionsTracker(country_iso_code="ESP",
                                   experiment_id=f"global_{model_id}_{map_name}",
                                   measure_power_secs=3600,
                                   log_level=logging.WARNING,
                                   output_dir=str(save_path),
                                   output_file=emissions_filename)


def main(argv):
    FLAGS = flags.FLAGS
    mode = FLAGS.mode
    config_files = FLAGS.config_files

    _CONFIG.read(config_files)
    flags.FLAGS(["runner"])

    _setup_logging(_CONFIG.get(mode, "log_file"))
    logger = MainLogger.get()
    logger.info(_CONFIG.get(mode, "model_id"))
    with io.StringIO() as ss:
        _CONFIG.write(ss)
        ss.seek(0) # rewind
        logger.info(f"Running with config\n{ss.read()}")

    save_path = Path(_CONFIG.get(mode, "save_path"))
    start_marker_file = save_path / f"_01_{mode}_start"
    end_marker_file = save_path / f"_02_{mode}_end"

    SC2_CONFIG["visualize"] = _CONFIG.getboolean(mode, "visualize")

    map_name = _CONFIG.get(mode, "map")
    map_config = MAP_CONFIGS[map_name]

    main_agent = _get_agent(mode)
    if len(map_config["players"]) > 1:
        opponent = _get_agent(mode + ".opponent")
        opponent.logger.setLevel(logging.WARNING)
        other_agents = [opponent]
    else:
        other_agents = []

    tracker = _get_tracker(mode)

    start_marker_file.touch(exist_ok=True)

    if mode == "train":
        finished = _train(main_agent, other_agents, tracker)
    else:
        finished = _exploit(main_agent, other_agents)

    end_marker_file.touch(exist_ok=True)

    return int(not finished)


if __name__ == "__main__":
    flags.DEFINE_enum("mode", default=None, required=True, enum_values=["train", "exploit"], help="What to mode to run, train or exploit.")
    flags.DEFINE_list("config_files", default=None, required=True, help="List of configuration files to use.")
    app.run(main)