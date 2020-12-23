cd /export/scratch/robert/survae_flows

# Load defaults for all experiments
#source /export/scratch/robert/survae_flows/experiments/manifold/scripts/experiment_config.sh
epochs=2
max_grad_norm=1.0
warmup=1
exponential_lr=False

dataset=mnist
batch_size=128
device=cuda

trainable_sigma=True
latent_size=196

num_scales=2
num_steps=1
dequant=uniform
densenet_blocks=1
densenet_channels=32
densenet_depth=1
densenet_growth=8


for seed in 101 #102 103
do

    printf "\n----------------------Seed ${seed}----------------------\n"

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
    
    printf "Linear Analytically NDP Flow Done\n"

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
    
    printf "Linear Stochastically NDP Flow Done\n"

    # Linear VAE
    python experiments/manifold/train.py --dataset ${dataset} \
           --linear False \
           --stochastic_elbo True \
           --pooling none \
           --vae_activation none \
           --vae_hidden_units 256 \
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

    printf "Linear VAE (256) Trained Done\n"


    # Non-Linear VAE
    python experiments/manifold/train.py --dataset ${dataset} \
           --linear False \
           --stochastic_elbo True \
           --pooling none \
           --vae_activation relu \
           --vae_hidden_units 512 256 \
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

    printf "ReLU VAE (512, 256) Done\n"


    # Max Pooling VAE
    python experiments/manifold/train.py --dataset ${dataset} \
           --linear False \
           --stochastic_elbo True \
           --pooling max \
           --vae_activation relu \
           --vae_hidden_units 128 \
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

    printf "Max Pooling (128) Done\n"

done


echo "Job complete"
