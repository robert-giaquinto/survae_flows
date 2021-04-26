#
# ------------- Variables shared across experiments -------------
#
epochs=300
optimizer=adamax
max_grad_norm=2.0
warmup=10000
exponential_lr=True
early_stop=50
eval_every=5

rho_init="decreasing"

dataset=svhn
num_bits=8
augmentation=neta

dequant=flow
dequant_steps=2
dequant_context=32

num_scales=3
num_steps=3

actnorm=True
augment_size=3
checkerboard_scales="0"

coupling_network=transformer
coupling_blocks=4
coupling_channels=64
coupling_dropout=0.0
coupling_mixtures=4

conditional_channels=""
lowres_upsampler_channels="32 64"
lowres_encoder_channels=32
lowres_encoder_blocks=1
lowres_encoder_depth=2
compression_ratio="0.25 0.33"
