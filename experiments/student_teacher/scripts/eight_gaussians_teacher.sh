#! /bin/bash

cd /export/scratch/robert/survae_flows


time python experiments/student_teacher/train_teacher.py \
     --dataset eight_gaussians \
     --device cpu \
     --epochs 10 \
     --num_flows 8 \
     --hidden_units 200 100 \
     --base_dist normal \
     --scale_fn softplus \
     --range_flow softplus \
     --clim 0.15 \
     --name eight_gaussians_abs_normal_teacher \
    ;
