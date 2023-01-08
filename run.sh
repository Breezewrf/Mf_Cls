#!/bin/bash
#SBATCH --job-name=Msfusion
#SBATCH --output=MSFusion.out
#SBATCH -e  log.out
#SBATCH --gres=gpu:1
#SBATCH -w node3

source /home/rfwei2/wrf/Mf_Cls/env/bin/activate
python3 train.py --amp --epochs 256 -l 1e-8 -b 1 --model msf --branch 2 --seed 57749867
python3 train.py --amp --epochs 256 -l 1e-8 -b 1 --model unet --branch 1 --seed 57749867
python3 train.py --amp --epochs 256 -l 1e-8 -b 1 --model unetpp --branch 1 --seed 57749867
