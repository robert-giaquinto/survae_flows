#! /bin/bash

cd /export/scratch/robert/survae_flows

# activate virtual environment
#source ./venv/bin/activate
# python --version
# pip freeze

# Load defaults for all experiments
source /export/scratch/robert/survae_flows/experiments/manifold/scripts/experiment_config.sh

printf "\nTraining NDP Flows with base arguments:\n"
cat /export/scratch/robert/survae_flows/experiments/manifold/scripts/experiment_config.sh

# Configurable parameters
gaussian_mid=False
epochs=100

# Fixed parameters
pooling=none
vae_activation=relu
linear=False
stochastic_elbo=True


for latent_sz in 196 392 588 784
do
    
    for seed in 1 #2 3
    do

        printf "\n\n----------------------Seed ${seed}, Latent Size ${latent_sz}----------------------\n\n"

        python experiments/manifold/train.py --dataset ${dataset} \
               --linear ${linear} \
               --stochastic_elbo ${stochastic_elbo} \
               --pooling ${pooling} \
               --vae_activation ${vae_activation} \
               --vae_hidden_units ${vae_hidden_units} \
               --gaussian_mid ${gaussian_mid} \
               --batch_size ${batch_size} \
               --epochs ${epochs} \
               --warmup ${warmup} \
               --exponential_lr ${exponential_lr} \
               --max_grad_norm ${max_grad_norm} \
               --latent_size ${latent_sz} \
               --num_scales ${num_scales} \
               --num_steps ${num_steps} \
               --dequant ${dequant} \
               --densenet_blocks ${densenet_blocks} \
               --densenet_channels ${densenet_channels} \
               --densenet_depth ${densenet_depth} \
               --densenet_growth ${densenet_growth} \
               --device ${device} \
               --seed ${seed}
    done
done


echo "Job complete"
