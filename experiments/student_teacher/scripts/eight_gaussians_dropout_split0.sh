#! /bin/bash

cd /export/scratch/robert/survae_flows


teacher="/export/scratch/robert/survae_flows/experiments/student_teacher/log/Teacher/eight_gaussians/abs_flows8_hidden200_100_affine/adam_lr1e-03/seed0/eight_gaussians_abs_normal_teacher"

time python experiments/student_teacher/train_baseline.py \
     --device cpu \
     --baseline dropout \
     --hidden_units 256 \
     --cond_trans split0 \
     --epochs 10 \
     --train_samples 50000 \
     --test_samples 50000 \
     --num_samples 50000 \
     --clim 1.0 \
     --name checkerboard_dropout_split0 \
     --teacher_model ${teacher} \
    ;
