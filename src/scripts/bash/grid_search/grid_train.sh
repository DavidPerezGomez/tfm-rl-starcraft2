#!/bin/bash

if [ -z "${SRC_DIR}" ]; then echo "Environmental variable SRC_DIR needs to be configured. Check the README for more info."; exit 1; fi

cd "${SRC_DIR}"

conf_list=( 01 02 03 04 05 06 07 )

for i in ${conf_list[@]};  do

  CONFIG_FILES="conf/default.ini, conf/attack_manager.ini, conf/grid_search/${i}.ini"

  python "${SRC_DIR}"/new_runner.py \
        --mode "train" \
        --config_files "${CONFIG_FILES}"

  python "${SRC_DIR}"/new_runner.py \
        --mode "exploit" \
        --config_files "${CONFIG_FILES}"

done