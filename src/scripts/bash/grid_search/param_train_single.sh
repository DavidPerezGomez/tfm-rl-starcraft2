#!/bin/bash

if [ -z "${SRC_DIR}" ]; then echo "Environmental variable SRC_DIR needs to be configured. Check the README for more info."; exit 1; fi
if [ -z "${MODELS_DIR}" ]; then echo "Environmental variable MODELS_DIR needs to be configured. Check the README for more info."; exit 1; fi

MODEL_SUBDIR=$1

AGENT_TYPE="single"
AGENT_ALGORITHM="dqn"
#AGENT_SUBTYPE="army_recruit_manager"
REWARD_MODE="score"
SCORE_METHOD="get_game_score_delta"

#AGENT_KEY="${AGENT_TYPE}.${AGENT_ALGORITHM}.${AGENT_SUBTYPE}"
#MODEL_ID="${AGENT_TYPE}_${AGENT_ALGORITHM}_${AGENT_SUBTYPE}_${REWARD_MODE}"

AGENT_KEY="${AGENT_TYPE}.${AGENT_ALGORITHM}"
MODEL_ID="${AGENT_TYPE}_${AGENT_ALGORITHM}_${REWARD_MODE}"

EXPERIMENT_DIR="${MODELS_DIR}/${MODEL_SUBDIR}"
MODEL_DIR="${EXPERIMENT_DIR}/${MODEL_ID}"
LOG_SUFFIX=""

MAP=Simple64
TRAIN_EPISODES=200
LEARNING_RATE_MILESTONES="120,160,180"
LEARNING_RATE=$2
GAMMA=$3

MAIN_NETWORK_UPDATE_FREQUENCY=1 # steps
TARGET_NETWORK_SYNC_FREQUENCY=75 # steps
TARGET_NETWORK_SYNC_MODE="soft" # soft
TARGET_NETWORK_UPDATE_TAU=0.1 # if soft

EPSILON=0.99
EPSILON_DECAY=0.972
MIN_EPSILON=0.01

DQN_SIZE="extra_large" # extra_small, small, medium, large, extra_large
MEMORY_SIZE=100000
BURN_IN=10000
BATCH_SIZE=1024

mkdir -p "${MODEL_DIR}"

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
      --dqn_size ${DQN_SIZE} \
      --memory_size ${MEMORY_SIZE} \
      --burn_in ${BURN_IN} \
      --batch_size ${BATCH_SIZE} \
      --action_masking \
      --reward_mode ${REWARD_MODE} \
      --score_method ${SCORE_METHOD} \
      --models_path "${EXPERIMENT_DIR}" \
      --log_file "training${LOG_SUFFIX}.log" \
      --buffer_file "buffer.pkl" \
      --save_frequency_episodes 10 \
      --novisualize \
      2>&1 | tee "${MODEL_DIR}"/${MAP}${LOG_SUFFIX}.log

touch "${MODEL_DIR}"/_02_training_done_${TRAIN_EPISODES}_ep