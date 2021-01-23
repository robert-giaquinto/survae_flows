#
# ------------- Variables shared across experiments -------------
#
device=cuda
dataset=mnist

# training parameters
epochs=300
batch_size=128
max_grad_norm=1.0
optimizer=adamax
warmup=5000
exponential_lr=True
annealing_schedule=25
early_stop=15
eval_every=5
check_every=10

# VAE parameters
trainable_sigma=True
vae_hidden_units="512 256"

# variational dequantization
dequant=flow
dequant_steps=2
dequant_context=8

# model architecture
num_scales=2
num_steps=8

# coupling layer network settings
coupling_network=transformer
coupling_blocks=2
coupling_channels=64
coupling_depth=1
coupling_growth=4
coupling_mixtures=16
coupling_dropout=0.2
