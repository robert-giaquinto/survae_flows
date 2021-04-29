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

dataset=celeba128
num_bits=5
augmentation=eta

num_scales=3
num_steps=4
checkerboard_scales="0"

coupling_network=transformer
coupling_blocks=12
coupling_channels=128
coupling_dropout=0.0
coupling_mixtures=4
actnorm=True

conditional_channels=""
lowres_upsampler_channels="64 64 128 128"
lowres_encoder_channels=64
lowres_encoder_blocks=2
lowres_encoder_depth=3
compression_ratio="0.0 0.5"

augment_size=4
augment_steps=4
augment_context=32
augment_blocks=4

dequant=flow
dequant_steps=4
dequant_context=32
dequant_blocks=4

