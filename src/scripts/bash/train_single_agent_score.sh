#!/bin/bash

EXPERIMENT_NAME="2024-09-12"

SCRIPTS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export SRC_DIR=$(realpath "$(dirname "$(dirname "${SCRIPTS_DIR}")")")
export BASE_MODELS_DIR=$(realpath "$(dirname "${SRC_DIR}")/models")
export BUFFERS_DIR="${BASE_MODELS_DIR}/buffers"

export AGENT_TYPE="single"
export AGENT_ALGORITHM="dqn"
export REWARD_METHOD="score"

export BASE_MODEL_ID=single_dqn
export MAP=Simple64

export TRAIN_EPISODES=100
export EPSILON_DECAY=0.98 # 300 EP
export LOG_SUFFIX="_${EXPERIMENT_NAME}"
export LEARNING_RATE_MILESTONES="40 70 90"
export LEARNING_RATE=0.001
export DQN_SIZE="large" # extra_small, small, medium, large, extra_large
export MEMORY_SIZE=100000
export BURN_IN=10000
export BATCH_SIZE=512

AGENT_KEY="${AGENT_TYPE}.${AGENT_ALGORITHM}"
MODELS_DIR="${BASE_MODELS_DIR}/${EXPERIMENT_NAME}"

# BUFFER_FILE="${BUFFERS_DIR}/${MAP_ID}_buffer.pkl"
MODEL_ID="${BASE_MODEL_ID}_${REWARD_METHOD}"

MODEL_DIR="${MODELS_DIR}/${MODEL_ID}"

mkdir -p "${MODEL_DIR}"

echo "Training ${MAP}"
touch "${MODEL_DIR}"/_01_training_start_${TRAIN_EPISODES}_ep

python "${SRC_DIR}"/runner.py \
    --agent_key "${AGENT_KEY}" \
    --map_name "${MAP}" \
    --num_episodes ${TRAIN_EPISODES} \
    --log_file "${MODEL_DIR}"/training${LOG_SUFFIX}.log \
    --model_id ${MODEL_ID} \
    --models_path "${MODELS_DIR}" \
    --epsilon_decay ${EPSILON_DECAY} \
    --lr_milestones "${LEARNING_RATE_MILESTONES}" \
    --lr ${LEARNING_RATE} \
    --dqn_size ${DQN_SIZE} \
    --memory_size ${MEMORY_SIZE} \
    --burn_in ${BURN_IN} \
    --batch_size ${BATCH_SIZE} \
    --reward_method ${REWARD_METHOD} 2>&1 | tee "${MODEL_DIR}"/${MAP}${LOG_SUFFIX}.log

touch "${MODEL_DIR}"/_02_training_done_${TRAIN_EPISODES}_ep
