#!/bin/bash

echo "Training multi-agent"
./src/scripts/bash/multi/attack_manager/train_attack_manager.sh 2>&1 | tee train_attack_manger.log
./src/scripts/bash/multi/base_manager/train_base_manager.sh 2>&1 | tee train_attack_manger.log
./src/scripts/bash/multi/recruit_manager/train_recruit_manager_random.sh 2>&1 | tee train_recruit_manager_random.log
./src/scripts/bash/multi/recruit_manager/train_recruit_manager.sh 2>&1 | tee train_recruit_manager.log
./src/scripts/bash/multi/game_manager/game_manager_fine_tune.sh 2>&1 | tee game_manager_fine_tune.log
./src/scripts/bash/multi/game_manager/game_manager.sh 2>&1 | tee game_manager.log

echo "Training single agent"
./src/scripts/bash/single/train_single_agent.sh 2>&1 | tee train_single_agent.log
