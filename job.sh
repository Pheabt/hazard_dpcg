#!/bin/bash
#SBATCH --job-name              procgen
#SBATCH --time                  24:00:00
#SBATCH --cpus-per-task         1       #maximum cpu limit for each v100 GPU is 6 , each a100 GPU is 8
#SBATCH --gres                  gpu:1
#SBATCH --mem                   200G      #maximum memory limit for each v100 GPU is 90G , each a100 GPU is 40G
#SBATCH --output                output.txt
#SBATCH --partition             h800_batch

source ~/.bashrc
source activate ha
python train.py  --env_name miner  --exp_name ppo  --use_which_gae fixed  --flag gamma999_nl0_tl0