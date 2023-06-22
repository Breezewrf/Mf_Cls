#!/bin/bash
#SBATCH --job-name=msfusion
#SBATCH --output=msfusion_adamw.out
########--output=msfusion_adamw.out
########--msfusion_adamw_error(patience=60).out
#SBATCH -e  msfusion_adamw_error(patience=60).out
#SBATCH --gres=gpu:1
#SBATCH -w node1

source /home/rfwei2/wrf/Mf_Cls/env/bin/activate
# python3 train.py --amp --epochs 256 -l 3e-4 -b 4 --model msf --branch 2 --seed 57749867 --aug 0 --desc AdamW
# python3 train.py --amp --epochs 256 -l 1e-8 -b 1 --model msf --branch 2 --seed 57749867 --aug 0 --desc 2
# python3 train.py --amp --epochs 256 -l 1e-8 -b 1 --model msf --branch 2 --seed 57749867 --aug 0 --desc 3

# testing
# python3 test.py --model msf --branch 2 --load /home/rfwei2/wrf/Mf_Cls/checkpoints/checkpoint_epoch200.pth

# Classification
# train 1
python train_cls.py --model resmsf
# train 2
python train_cls.py --model resmsf_gain --load /media/breeze/dev/Mf_Cls/checkpoints/classification/stream3-epochs[200]-bs[8]-lr[3e-05]-c2-ds[prostatex]-modal[pre_fuse]-focal-fuse4-exp1000-v2/checkpoint_epoch200.pth --learning-rate 3e-7

# test 1
python test_cls.py --model resmsf --load /media/breeze/dev/Mf_Cls/checkpoints/classification/stream3-epochs[200]-bs[16]-lr[3e-05]-c2-ds[prostatex]-modal[pre_fuse]-focal-fuse4-exp10/checkpoint_epoch100.pth --test_mode True
# test 2
python test_cls.py --model resmsf_gain --load /media/breeze/dev/Mf_Cls/checkpoints/classification/gain-stream3-epochs[200]-bs[8]-lr[3e-07]-c2-ds[prostatex]-modal[pre_fuse]-focal-fuse4-exp1000-v2/checkpoint_epoch50.pth --test_mode True

# Segmentation
python test.py --branch 3 --load /media/breeze/dev/Mf_Cls/checkpoints/unified/epochs[500]-bs[4]-lr[0.0003]-c2-ds[prostatex]-modal[pre_fuse]-focal-deep-sigmoid/best_epoch.pth --deep True --use_cam False
# 6 stream
python test.py --branch 6 --load /media/breeze/dev/Mf_Cls/checkpoints/unified/streams6-epochs[500]-bs[2]-lr[0.0003]-c2-ds[prostatex]-modal[pre_fuse]-focal-sigmoid/best_epoch.pth --batch-size 4 --use_cam True
