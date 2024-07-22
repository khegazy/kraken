#!/bin/bash

WANDB_MODE=disabled python3 scOT/train.py \
    --config configs/run.yaml \
    --checkpoint_path ./checkpoints/ \
    --data_path /pscratch/sd/k/khegazy/datasets/pdes/test/ \
    #--replace_embedding_recovery <SET ONLY IF EMBED/RECOVERY NEEDS TO BE REPLACED>
    #--finetune_from <PRETRAINED_MODEL> \
    #--wandb_run_name test \
    #--wandb_project_name test \