# Variables shared across experiments
dataset=mnist
device=cuda

batch_size=128
max_grad_norm=1.0
epochs=50
warmup=3
exponential_lr=False

trainable_sigma=True
latent_size=196
vae_hidden_units="512 256"

num_scales=2
num_steps=3
dequant=uniform
densenet_blocks=1
densenet_channels=64
densenet_depth=2
densenet_growth=16
