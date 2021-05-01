#
# ------------- Variables shared across experiments -------------
#
epochs=20
optimizer=adamax
max_grad_norm=2.0
warmup=20000
exponential_lr=True
early_stop=0
eval_every=1
check_every=1

dataset=celeba64
num_bits=5
augmentation=eta

num_scales=3
num_steps=4

actnorm=True
checkerboard_scales="0"

coupling_network=transformer
coupling_blocks=12
coupling_channels=96
coupling_dropout=0.0
coupling_mixtures=4

conditional_channels=""
lowres_upsampler_channels="32 32 64 64"
lowres_encoder_channels=64
lowres_encoder_blocks=2
lowres_encoder_depth=4
compression_ratio="0.25 0.33"

augment_size=4
augment_steps=4
augment_context=32
augment_blocks=5

dequant=flow
dequant_steps=4
dequant_context=32
dequant_blocks=5

