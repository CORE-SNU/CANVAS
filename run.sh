#!/usr/bin/env bash

predictors=(linear gp eigen traj koopcast socialvae socialstgcnn)
datasets=(ETH Hotel Univ Zara01 Zara02)

for predictor in "${predictors[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "Running with predictor=${predictor}, dataset=${dataset}"
    python simulation_video_spectrum_mpl_contCI.py --predictor "$predictor" --dataset "$dataset"
  done
done