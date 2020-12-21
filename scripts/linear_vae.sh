cd /export/scratch/robert/survae_flows

# activate virtual environment
#source ./venv/bin/activate
# python --version
# pip freeze

# Load defaults for all experiments


# variables specific to this experiment
epochs=50
dataset=mnist
batch_size=128

trainable_sigma=True
latent_size=196
num_scales=2
num_steps=3
dequant=uniform
densenet_blocks=1
densenet_channels=64
densenet_depth=2
densenet_growth=16


for seed in 101 102 103
do
	python experiments/manifold/train.py --linear False \
		--stochastic_elbo True \
		--pooling none \
		--vae_hidden_units 512 256 \
		--vae_activation none \
		--dataset ${dataset} \
		--batch_size ${batch_size} \
		--epochs ${epochs} \
		--trainable_sigma ${trainable_sigma} \
		--latent_size ${latent_size} \
		--num_scales ${num_scales} \
		--num_steps ${num_steps} \
		--dequant ${dequant} \
		--densenet_blocks ${densenet_blocks} \
		--densenet_channels ${densenet_channels} \
		--densenet_depth ${densenet_depth} \
		--densenet_growth ${densenet_growth} \
		--seed ${seed} \
		--device cuda
done

echo "Job complete"
