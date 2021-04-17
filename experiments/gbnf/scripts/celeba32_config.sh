#
# ------------- Variables shared across experiments -------------
#

# optimization and training defaults
epochs=20
optimizer=adamax
max_grad_norm=1.0
warmup=10000
exponential_lr=True
early_stop=0
eval_every=1

# dataset and augmentations to images (flip horizontal, edge padding, small translations, and centering)
dataset=celeba32
augmentation=eta
num_bits=8

# flow architecture defaults
num_scales=3
num_steps=4
# compressive and sliced flows reduce dimensionality by half at each scale:
compression_ratio="0.5"

actnorm=True
augment_size=0

# coupling architecture
coupling_network=transformer
coupling_blocks=8
coupling_channels=96
coupling_dropout=0.0
coupling_mixtures=4

# dequanitization
dequant=flow
dequant_steps=4
dequant_context=32

# conditional actnorm, 1x1 inv conv, and context initializer defaults
conditional_channels=""
lowres_upsampler_channels="32 64"
lowres_encoder_channels=32
lowres_encoder_blocks=1
lowres_encoder_depth=3

# boosting defaults (when used)
rho_init="decreasing"
