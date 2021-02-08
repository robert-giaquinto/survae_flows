#! /bin/bash

cd /export/scratch/robert/survae_flows

# activate virtual environment
#source ./venv/bin/activate
# python --version
# pip freeze

# Load defaults for all experiments
source /export/scratch/robert/survae_flows/experiments/manifold/scripts/cifar_config.sh

printf "\nTraining Flows with base arguments:\n"
cat /export/scratch/robert/survae_flows/experiments/manifold/scripts/cifar_config.sh


# Configurable parameters
vae_hidden_units="512 256"
vae_activation=relu
latent_size=384

seed=101
epochs=850
lr=0.001
batch_size=128

printf "\nCompressing Pretrained Model with VAE (${vae_activation}, hidden=${vae_hidden_units})\n"
printf "\n\n---------------------- Seed ${seed}, Latent Size ${latent_size} ----------------------\n\n"

python experiments/manifold/train_pretrained.py \
    --model            "/export/scratch/robert/survae_flows/experiments/manifold/log/cifar10_8bit/pool_flow/more/nonpool/" \
    --new_device       cuda \
    --amp              False \
    --freeze           True \
    --seed             ${seed} \
    --new_epochs       ${epochs} \
    --new_lr           ${lr} \
    --new_batch_size   ${batch_size} \
    --vae_activation   ${vae_activation} \
    --vae_hidden_units ${vae_hidden_units} \
    --latent_size      ${latent_size} \
    ;

echo "Job complete"
