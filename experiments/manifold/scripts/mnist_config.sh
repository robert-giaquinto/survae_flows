#
# ------------- Variables shared across experiments -------------
#
device=cuda
dataset=mnist

# training parameters
epochs=300
batch_size=32
max_grad_norm=1.0
optimizer=adamax
warmup=5000
exponential_lr=True
annealing_schedule=25
early_stop=15
eval_every=5

# VAE parameters
trainable_sigma=True
latent_size=196
vae_hidden_units="512 256"

# variational dequantization
dequant=flow
dequant_steps=1
dequant_context=8

# model architecture
num_scales=2
num_steps=4

# coupling layer network settings
coupling_network=transformer
coupling_blocks=1
coupling_channels=32
coupling_depth=1
coupling_growth=4
coupling_mixtures=8
coupling_dropout=0.2
