#!/bin/bash

if [ -z "${SRC_DIR}" ]; then echo "Environmental variable SRC_DIR needs to be configured. Check the README for more info."; exit 1; fi

cd "${SRC_DIR}"

CONFIG_FILES="conf/default.ini, conf/recruit_manage_explore.ini"

python "${SRC_DIR}"/runner.py \
      --mode "train" \
      --config_files "${CONFIG_FILES}"