#! /bin/bash

cd /export/scratch/robert/survae_flows

# activate virtual environment
#source ./venv/bin/activate
# python --version
# pip freeze

# Load defaults for all experiments
source /export/scratch/robert/survae_flows/experiments/manifold/scripts/mnist_config.sh

printf "\nTraining Flows with base arguments:\n"
cat /export/scratch/robert/survae_flows/experiments/manifold/scripts/mnist_config.sh

# Configurable parameters
base_distributions="n"

# Fixed parameters for this experiment
compression=vae
vae_activation=relu
linear=False
stochastic_elbo=True
latent_size=196


printf "\nFor compression=${vae}, vae_activation=${vae_activation}, linear=${linear}, coupling=${coupling_network} (${base_distributions}), and latent={latent_size}\n"

for num_steps in 0 2 4
do
    
    for seed in 101 #102 103
    do

        printf "\n\n---------------------- Seed ${seed}, Num Flow Steps ${num_steps} ----------------------\n\n"

        python experiments/manifold/train.py \
               --device             ${device} \
               --seed               ${seed} \
               --dataset            ${dataset} \
               \
               --compression        ${compression} \
               --linear             ${linear} \
               --stochastic_elbo    ${stochastic_elbo} \
               --base_distributions ${base_distributions} \
               \
               --optimizer          ${optimizer} \
               --batch_size         ${batch_size} \
               --epochs             ${epochs} \
               --warmup             ${warmup} \
               --exponential_lr     ${exponential_lr} \
               --max_grad_norm      ${max_grad_norm} \
               --annealing_schedule ${annealing_schedule} \
               --early_stop         ${early_stop} \
               --eval_every         ${eval_every} \
               --check_every        ${check_every} \
               \
               --latent_size        ${latent_size} \
               --vae_activation     ${vae_activation} \
               --vae_hidden_units   ${vae_hidden_units} \
               \
               --num_scales         ${num_scales} \
               --num_steps          ${num_steps} \
               \
               --dequant            ${dequant} \
               --dequant_steps      ${dequant_steps} \
               --dequant_context    ${dequant_context} \
               \
               --coupling_network   ${coupling_network} \
               --coupling_blocks    ${coupling_blocks} \
               --coupling_channels  ${coupling_channels} \
               --coupling_depth     ${coupling_depth} \
               --coupling_growth    ${coupling_growth} \
               --coupling_dropout   ${coupling_dropout} \
               --coupling_mixtures  ${coupling_mixtures} \
               ;
        
    done
done


echo "Job complete"
