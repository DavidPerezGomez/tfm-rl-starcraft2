#!/bin/bash

./base_manager/train_base_manager.sh
./recruit_manager/train_recruit_manager.sh
./attack_manager/train_attack_manager.sh

./base_manager/exploit_base_manager.sh
./recruit_manager/exploit_recruit_manager.sh
./attack_manager/exploit_attack_manager.sh