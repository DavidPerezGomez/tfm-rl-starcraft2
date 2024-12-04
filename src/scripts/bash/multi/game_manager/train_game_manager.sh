#!/bin/bash

if [ -z "${SRC_DIR}" ]; then echo "Environmental variable SRC_DIR needs to be configured. Check the README for more info."; exit 1; fi
if [ -z "${MODELS_DIR}" ]; then echo "Environmental variable MODELS_DIR needs to be configured. Check the README for more info."; exit 1; fi

MODEL_SUBDIR="gm_2024-12-02_1"

AGENT_TYPE="multi"
AGENT_ALGORITHM="dqn"
AGENT_SUBTYPE="game_manager"
REWARD_MODE="score"
SCORE_MODE="get_game_score_delta"

AGENT_KEY="${AGENT_TYPE}.${AGENT_ALGORITHM}.${AGENT_SUBTYPE}"
MODEL_ID="${AGENT_TYPE}_${AGENT_ALGORITHM}_${AGENT_SUBTYPE}_${REWARD_MODE}"

EXPERIMENT_DIR="${MODELS_DIR}/${MODEL_SUBDIR}"
MODEL_DIR="${EXPERIMENT_DIR}/${MODEL_ID}"
LOG_SUFFIX=""

MAP=Simple64
TRAIN_EPISODES=50
LEARNING_RATE_MILESTONES="30,40,45"
LEARNING_RATE=0.003
GAMMA=0.95

MAIN_NETWORK_UPDATE_FREQUENCY=1 # steps
TARGET_NETWORK_SYNC_FREQUENCY=100 # steps
TARGET_NETWORK_SYNC_MODE="soft" # hard, soft
TARGET_NETWORK_UPDATE_TAU=0.1 # if soft

GM_TIME_DISPLACEMENT=5

EPSILON=0.99
EPSILON_DECAY=0.987 # ~42 episodes until minimum
MIN_EPSILON=0.01

DQN_SIZE="medium" # extra_small, small, medium, large, extra_large
MEMORY_SIZE=10000
BURN_IN=64 # ~6 episodes
BATCH_SIZE=32

mkdir -p "${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"/base_manager/
mkdir -p "${MODEL_DIR}"/army_recruit_manager/
mkdir -p "${MODEL_DIR}"/army_attack_manager/

# Copy sub-agents
BASE_MANAGER_MODEL_DIR="${MODELS_DIR}"/best_32/multi_dqn_base_manager_reward
ARMY_RECRUIT_MANAGER_MODEL_DIR="${MODELS_DIR}"/best_32/multi_dqn_army_recruit_manager_score
ARMY_ATTACK_MANAGER_MODEL_DIR="${MODELS_DIR}"/best_32/multi_dqn_army_attack_manager_score

cp "${BASE_MANAGER_MODEL_DIR}"/*.pt "${MODEL_DIR}"/base_manager/
cp "${ARMY_RECRUIT_MANAGER_MODEL_DIR}"/*.pt "${MODEL_DIR}"/army_recruit_manager/
cp "${ARMY_ATTACK_MANAGER_MODEL_DIR}"/*.pt "${MODEL_DIR}"/army_attack_manager/

echo "Training ${AGENT_KEY} on ${MAP}"
touch "${MODEL_DIR}"/_01_training_start_${TRAIN_EPISODES}_ep

python "${SRC_DIR}"/runner.py \
      --agent_key "${AGENT_KEY}" \
      --model_id "${MODEL_ID}" \
      --map_name "${MAP}" \
      --num_episodes ${TRAIN_EPISODES} \
      --lr_milestones "${LEARNING_RATE_MILESTONES}" \
      --lr ${LEARNING_RATE} \
      --gamma ${GAMMA} \
      --main_network_update_frequency ${MAIN_NETWORK_UPDATE_FREQUENCY} \
      --target_network_sync_frequency ${TARGET_NETWORK_SYNC_FREQUENCY} \
      --target_sync_mode ${TARGET_NETWORK_SYNC_MODE} \
      --update_tau ${TARGET_NETWORK_UPDATE_TAU} \
      --epsilon ${EPSILON} \
      --epsilon_decay ${EPSILON_DECAY} \
      --min_epsilon ${MIN_EPSILON} \
      --gm_time_displacement ${GM_TIME_DISPLACEMENT} \
      --dqn_size ${DQN_SIZE} \
      --memory_size ${MEMORY_SIZE} \
      --burn_in ${BURN_IN} \
      --base_subagent_memory_size 10000 \
      --base_subagent_burn_in 32 \
      --recruit_subagent_memory_size 10000 \
      --recruit_subagent_burn_in 32 \
      --attack_subagent_memory_size 10000 \
      --attack_subagent_burn_in 32 \
      --batch_size ${BATCH_SIZE} \
      --action_masking \
      --reward_mode ${REWARD_MODE} \
      --score_method ${SCORE_MODE} \
      --fine_tune \
      --base_subagent_reward_mode score \
      --base_subagent_score_method get_mineral_count_delta \
      --recruit_subagent_reward_mode score \
      --recruit_subagent_score_method get_army_spending_delta \
      --attack_subagent_reward_mode score \
      --attack_subagent_score_method get_health_difference_score_delta \
      --models_path "${EXPERIMENT_DIR}" \
      --log_file "training${LOG_SUFFIX}.log" \
      --buffer_file "buffer.pkl" \
      --load_networks_only \
      --save_frequency_episodes 1 \
      --visualize \
      2>&1 | tee "${MODEL_DIR}"/${MAP}${LOG_SUFFIX}.log

touch "${MODEL_DIR}"/_02_training_done_${TRAIN_EPISODES}_ep