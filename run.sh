#!/bin/bash
#SBATCH --job-name=Msfusion
#SBATCH --output=MSFusion.out
#SBATCH -e  log.out
#SBATCH --gres=gpu:1
#SBATCH -w node3

source /home/rfwei2/wrf/Mf_Cls/env/bin/activate
python3 train.py --amp --epochs 512 -l 1e-8 -b 1 --model msf --seed 57749867
python3 train.py --amp --epochs 512 -l 1e-8 -b 1 --model msf --seed 57749867
python3 train.py --amp --epochs 512 -l 1e-8 -b 1 --model msf --seed 57749867
python3 train.py --amp --epochs 512 -l 1e-8 -b 1 --model msf --seed 57749867
