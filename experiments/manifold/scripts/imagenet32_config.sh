#
# ------------- Variables shared across experiments -------------
#
device=cuda
dataset=imagenet32
augmentation=none

# training parameters
epochs=25
batch_size=64
max_grad_norm=1.0
optimizer=adamax
warmup=5000
exponential_lr=True
annealing_schedule=25
early_stop=10
eval_every=1
check_every=1

# VAE parameters
trainable_sigma=True
vae_hidden_units="512 256"
vae_activation=relu

# variational dequantization
dequant=flow
dequant_steps=4
dequant_context=32

# model architecture
num_scales=2
num_steps=12

# coupling layer network settings
coupling_network=transformer
coupling_blocks=10
coupling_channels=96
coupling_depth=1
coupling_growth=4
coupling_mixtures=32
coupling_dropout=0.2
