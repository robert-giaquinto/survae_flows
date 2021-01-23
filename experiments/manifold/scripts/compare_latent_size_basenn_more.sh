#! /bin/bash

cd /export/scratch/robert/survae_flows

# Re run 196
time python experiments/manifold/train_more.py \
     --model              /export/scratch/robert/survae_flows/experiments/manifold/log/mnist_8bit/NDP_VAE_512_256_Flow_latent196_basecn_scales2_steps8_transformer/adamax_lr001/seed101/2021-01-14_13-11-52 \
     --new_epochs         300 \
     --base_distributions nn \
    ;

# Re run 392
time python experiments/manifold/train_more.py \
     --model              /export/scratch/robert/survae_flows/experiments/manifold/log/mnist_8bit/NDP_VAE_512_256_Flow_latent392_basecn_scales2_steps8_transformer/adamax_lr001/seed101/2021-01-16_16-19-07 \
     --new_epochs         300 \
     --base_distributions nn \
    ;

# Re run 588
time python experiments/manifold/train_more.py \
     --model              /export/scratch/robert/survae_flows/experiments/manifold/log/mnist_8bit/NDP_VAE_512_256_Flow_latent588_basecn_scales2_steps8_transformer/adamax_lr001/seed101/2021-01-18_06-19-39 \
     --new_epochs         300 \
     --base_distributions nn \
    ;

# Re run 784
#time python experiments/manifold/train_more.py \
#     --model              /export/scratch/robert/survae_flows/experiments/manifold/log/mnist_8bit/NDP_VAE_512_256_Flow_latent784_basecn_scales2_steps8_transformer/adamax_lr001/seed101/ \
#     --new_epochs         300 \
#     --base_distributions nn \
#    ;


echo "Job complete"
