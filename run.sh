#!/bin/bash
#SBATCH --job-name=Msfusion
#SBATCH --output=MSFusion.out
#SBATCH -e  error.out
#SBATCH --gres=gpu:1
#SBATCH -w node3
nvidia-smi
python3 train.py --amp --epochs 100 -l 2e-7 --model msf --seed 59363379
