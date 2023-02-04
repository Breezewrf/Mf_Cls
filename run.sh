#!/bin/bash
#SBATCH --job-name=msfusion
#SBATCH --output=msfusion_adamw.out
########--output=msfusion_adamw.out
########--msfusion_adamw_error(patience=60).out
#SBATCH -e  msfusion_adamw_error(patience=60).out
#SBATCH --gres=gpu:1
#SBATCH -w node1

source /home/rfwei2/wrf/Mf_Cls/env/bin/activate
python3 train.py --amp --epochs 256 -l 3e-4 -b 4 --model msf --branch 2 --seed 57749867 --aug 0 --desc AdamW
# python3 train.py --amp --epochs 256 -l 1e-8 -b 1 --model msf --branch 2 --seed 57749867 --aug 0 --desc 2
# python3 train.py --amp --epochs 256 -l 1e-8 -b 1 --model msf --branch 2 --seed 57749867 --aug 0 --desc 3

# testing
# python3 test.py --model msf --branch 2 --load /home/rfwei2/wrf/Mf_Cls/checkpoints/checkpoint_epoch200.pth
