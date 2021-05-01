#
# ------------- Variables shared across experiments -------------
#
epochs=10
optimizer=adamax
max_grad_norm=2.0
warmup=20000
exponential_lr=True
early_stop=0
eval_every=1
check_every=1

dataset=imagenet32
num_bits=8
augmentation=none

num_scales=3
num_steps=4
checkerboard_scales="0"

coupling_network=transformer
coupling_blocks=16
coupling_channels=128
coupling_dropout=0.0
coupling_mixtures=8
actnorm=True

conditional_channels=""
lowres_upsampler_channels="32 32 64 64"
lowres_encoder_channels=64
lowres_encoder_blocks=2
lowres_encoder_depth=3
compression_ratio="0.25 0.33"

augment_size=3
augment_steps=8
augment_context=32
augment_blocks=8

dequant=flow
dequant_steps=4
dequant_context=32
dequant_blocks=5

