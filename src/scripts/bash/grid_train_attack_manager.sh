#!/bin/bash

main_list=( 1 25 50 )
target_list=( 100 250 500 )
tau_list=( 0.1 0.5 1 )
i=1

for fm in ${main_list[@]};  do
  for ft in ${target_list[@]};  do
    for tau in ${tau_list[@]};  do

      EXPERIMENT_NAME="2024-11-18_${i}"

      SCRIPTS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
      SRC_DIR=$(realpath "$(dirname "$(dirname "${SCRIPTS_DIR}")")")
      BASE_MODELS_DIR=$(realpath "$(dirname "${SRC_DIR}")/models")

      AGENT_TYPE="multi"
      AGENT_ALGORITHM="dqn"
      AGENT_SUBTYPE="army_attack_manager"
      REWARD_MODE="reward"

      AGENT_KEY="${AGENT_TYPE}.${AGENT_ALGORITHM}.${AGENT_SUBTYPE}"
      MODEL_ID="${AGENT_TYPE}_${AGENT_ALGORITHM}_${AGENT_SUBTYPE}_${REWARD_MODE}"

      EXPERIMENT_DIR="${BASE_MODELS_DIR}/${EXPERIMENT_NAME}"
      MODEL_DIR="${EXPERIMENT_DIR}/${MODEL_ID}"
      LOG_SUFFIX="_${EXPERIMENT_NAME}"

      MAP=DefeatBase
      TRAIN_EPISODES=150
      LEARNING_RATE_MILESTONES="75,125"
      LEARNING_RATE=0.001
      GAMMA=0.99

      MAIN_NETWORK_UPDATE_FREQUENCY=${fm} #1 # steps
      TARGET_NETWORK_SYNC_FREQUENCY=${ft} #300 # steps
      TARGET_NETWORK_UPDATE_TAU=${tau} #0.5 # if soft
      if [[ "${tau}" == "1" ]]; then
        TARGET_NETWORK_SYNC_MODE="hard" # soft
      else
        TARGET_NETWORK_SYNC_MODE="soft" # soft
      fi

      EPSILON=0.9
      EPSILON_DECAY=0.99
      MIN_EPSILON=0.01

      DQN_SIZE="small" # extra_small, small, medium, large, extra_large
      MEMORY_SIZE=10000
      BURN_IN=1000
      BATCH_SIZE=512

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
      --score_method ${SCORE_MODE} \
            --models_path "${EXPERIMENT_DIR}" \
            --log_file "training${LOG_SUFFIX}.log" \
            --buffer_file "buffer.pkl" \
            2>&1 | tee "${MODEL_DIR}"/${MAP}${LOG_SUFFIX}.log

      touch "${MODEL_DIR}"/_02_training_done_${TRAIN_EPISODES}_ep

      i=$(($i+1))
    done
  done
done