#!/bin/bash

#lr_list=( 0.001 0.005 )
#gamma_list=( 0.85 0.95 0.9995 )
#decay_list=( 0.89 0.832 )
#i=1

#for lr in ${lr_list[@]};  do
#  for d in ${decay_list[@]};  do
#    for g in ${gamma_list[@]};  do

  #      if [ $i -eq 1 ]; then echo "skipping"; continue; fi

      MODEL_SUBDIR="single_2024-12-02_6"

      ./param_train_single.sh $MODEL_SUBDIR 0.005 0.9995
      ./param_exploit_single.sh $MODEL_SUBDIR

#      i=$(($i+1))
#    done
#  done
#done