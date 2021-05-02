#! /bin/bash

cd /export/scratch/robert/survae_flows


time python experiments/student_teacher/train_teacher.py \
     --dataset checkerboard \
     --device cpu \
     --epochs 5 \
     --num_flows 4 \
     --hidden_units 200 100 \
     --base_dist uniform \
     --name abs_uniform_teacher \
    ;


    
