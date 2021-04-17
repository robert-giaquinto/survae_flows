#
# ------------- Variables shared across experiments -------------
#
epochs=300
optimizer=adamax
max_grad_norm=1.0
warmup=10000
exponential_lr=True
early_stop=75
eval_every=5

rho_init="decreasing"

# dataset and augmentations to images (flip horizontal, edge padding, small translations, and centering)
dataset=cifar10
num_bits=8
augmentation=eta

dequant=flow
dequant_steps=4
dequant_context=32

num_scales=3
num_steps=4

actnorm=True
augment_size=0

coupling_network=transformer
coupling_blocks=8
coupling_channels=96
coupling_dropout=0.0
coupling_mixtures=4

conditional_channels=""
lowres_upsampler_channels="32 64"
lowres_encoder_channels=32
lowres_encoder_blocks=1
lowres_encoder_depth=3
compression_ratio="0.5"
