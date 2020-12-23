cd /export/scratch/robert/survae_flows

# activate virtual environment
#source ./venv/bin/activate
# python --version
# pip freeze

# Load defaults for all experiments
source /export/scratch/robert/survae_flows/experiments/manifold/scripts/experiment_config.sh

printf "\nTraining NDP Flows with base arguments:\n"
cat /export/scratch/robert/survae_flows/experiments/manifold/scripts/experiment_config.sh


for seed in 101 #102 103
do

    printf "\n\n----------------------Seed ${seed}----------------------\n\n"

    # linear analytic
    python experiments/manifold/train.py --dataset ${dataset} \
           --linear True \
           --stochastic_elbo False \
           --batch_size ${batch_size} \
           --epochs ${epochs} \
           --warmup ${warmup} \
           --exponential_lr ${exponential_lr} \
           --max_grad_norm ${max_grad_norm} \
           --trainable_sigma ${trainable_sigma} \
           --latent_size ${latent_size} \
           --num_scales ${num_scales} \
           --num_steps ${num_steps} \
           --dequant ${dequant} \
           --densenet_blocks ${densenet_blocks} \
           --densenet_channels ${densenet_channels} \
           --densenet_depth ${densenet_depth} \
           --densenet_growth ${densenet_growth} \
           --device ${device} \
           --seed ${seed}
    
    printf "Linear Analytically NDP Flow Done\n\n"

    # linear stochastic
    python experiments/manifold/train.py --dataset ${dataset} \
           --linear True \
           --stochastic_elbo True \
           --batch_size ${batch_size} \
           --epochs ${epochs} \
           --warmup ${warmup} \
           --exponential_lr ${exponential_lr} \
           --max_grad_norm ${max_grad_norm} \
           --trainable_sigma ${trainable_sigma} \
           --latent_size ${latent_size} \
           --num_scales ${num_scales} \
           --num_steps ${num_steps} \
           --dequant ${dequant} \
           --densenet_blocks ${densenet_blocks} \
           --densenet_channels ${densenet_channels} \
           --densenet_depth ${densenet_depth} \
           --densenet_growth ${densenet_growth} \
           --device ${device} \
           --seed ${seed}
    
    printf "Linear Stochastically NDP Flow Done\n\n"

    # Linear VAE
    python experiments/manifold/train.py --dataset ${dataset} \
           --linear False \
           --stochastic_elbo True \
           --pooling none \
           --vae_activation none \
           --vae_hidden_units ${vae_hidden_units} \
           --batch_size ${batch_size} \
           --epochs ${epochs} \
           --warmup ${warmup} \
           --exponential_lr ${exponential_lr} \
           --max_grad_norm ${max_grad_norm} \
           --trainable_sigma ${trainable_sigma} \
           --latent_size ${latent_size} \
           --num_scales ${num_scales} \
           --num_steps ${num_steps} \
           --dequant ${dequant} \
           --densenet_blocks ${densenet_blocks} \
           --densenet_channels ${densenet_channels} \
           --densenet_depth ${densenet_depth} \
           --densenet_growth ${densenet_growth} \
           --device ${device} \
           --seed ${seed}

    printf "Linear VAE (${vae_hidden_units}) Trained Done\n\n"


    # Non-Linear VAE
    python experiments/manifold/train.py --dataset ${dataset} \
           --linear False \
           --stochastic_elbo True \
           --pooling none \
           --vae_activation relu \
           --vae_hidden_units ${vae_hidden_units} \
           --batch_size ${batch_size} \
           --epochs ${epochs} \
           --warmup ${warmup} \
           --exponential_lr ${exponential_lr} \
           --max_grad_norm ${max_grad_norm} \
           --trainable_sigma ${trainable_sigma} \
           --latent_size ${latent_size} \
           --num_scales ${num_scales} \
           --num_steps ${num_steps} \
           --dequant ${dequant} \
           --densenet_blocks ${densenet_blocks} \
           --densenet_channels ${densenet_channels} \
           --densenet_depth ${densenet_depth} \
           --densenet_growth ${densenet_growth} \
           --device ${device} \
           --seed ${seed}

    printf "ReLU VAE (${vae_hidden_units}) Done\n"


    # Max Pooling VAE
    python experiments/manifold/train.py --dataset ${dataset} \
           --linear False \
           --stochastic_elbo True \
           --pooling max \
           --vae_activation relu \
           --vae_hidden_units ${vae_hidden_units} \
           --batch_size ${batch_size} \
           --epochs ${epochs} \
           --warmup ${warmup} \
           --exponential_lr ${exponential_lr} \
           --max_grad_norm ${max_grad_norm} \
           --trainable_sigma ${trainable_sigma} \
           --latent_size ${latent_size} \
           --num_scales ${num_scales} \
           --num_steps ${num_steps} \
           --dequant ${dequant} \
           --densenet_blocks ${densenet_blocks} \
           --densenet_channels ${densenet_channels} \
           --densenet_depth ${densenet_depth} \
           --densenet_growth ${densenet_growth} \
           --device ${device} \
           --seed ${seed}

    printf "Max Pooling (${vae_hidden_units}) Done\n\n"

done


echo "Job complete"
