#! /bin/bash

cd /export/scratch/robert/survae_flows

teacher="/export/scratch/robert/survae_flows/experiments/student_teacher/log/Teacher/checkerboard/abs_flows4_hidden200_100_affine/adam_lr1e-03/seed0/abs_uniform_teacher"

time python experiments/student_teacher/train_baseline.py \
     --device cpu \
     --baseline dropout \
     --hidden_units 256 \
     --cond_trans quantize4 \
     --epochs 10 \
     --train_samples 50000 \
     --test_samples 50000 \
     --num_samples 50000 \
     --clim 1.0 \
     --name checkerboard_dropout_quantize4 \
     --teacher_model ${teacher} \
    ;
