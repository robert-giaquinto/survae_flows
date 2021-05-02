#! /bin/bash

cd /export/scratch/robert/survae_flows


teacher="/export/scratch/robert/survae_flows/experiments/student_teacher/log/Teacher/eight_gaussians/abs_flows8_hidden200_100_affine/adam_lr1e-03/seed0/eight_gaussians_abs_normal_teacher"

time python experiments/student_teacher/train_student.py \
     --device cpu \
     --num_flows 4 \
     --base_dist normal \
     --augment_size 2 \
     --hidden_units 200 100 \
     --teacher_model ${teacher} \
     --epochs 5 \
     --cond_trans split0 \
     --name eight_gaussians_aug_normal_split0 \
    ;

