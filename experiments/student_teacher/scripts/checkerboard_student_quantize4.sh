#! /bin/bash

cd /export/scratch/robert/survae_flows

teacher="/export/scratch/robert/survae_flows/experiments/student_teacher/log/Teacher/checkerboard/abs_flows4_hidden200_100_affine/adam_lr1e-03/seed0/abs_uniform_teacher"

time python experiments/student_teacher/train_student.py \
     --device cpu \
     --base_dist normal \
     --augment_size 2 \
     --hidden_units 200 100 \
     --teacher_model ${teacher} \
     --epochs 5 \
     --name checkerboard_aug_normal_quantize4 \
     --cond_trans quantize4 \
    ;

