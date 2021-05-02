#! /bin/bash

cd /export/scratch/robert/survae_flows

teacher="/export/scratch/robert/survae_flows/experiments/student_teacher/log/Teacher/eight_gaussians/abs_flows8_hidden200_100_affine/adam_lr1e-03/seed0/eight_gaussians_abs_normal_teacher"

time python experiments/student_teacher/train_baseline.py \
     --device cpu \
     --baseline gp \
     --cond_trans quantize4 \
     --train_samples 25000 \
     --test_samples 5000 \
     --num_samples 10000 \
     --clim 1.0 \
     --name checkerboard_gp_quantize4 \
     --teacher_model ${teacher} \
    ;
