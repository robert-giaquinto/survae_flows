#
# ------------- Variables shared across experiments -------------
#
epochs=20
optimizer=adamax
max_grad_norm=1.0
warmup=10000
exponential_lr=True
early_stop=0
eval_every=1

boosted_components=1
rho_init="decreasing"

sr_scale_factor=8

dataset=celeba32
batch_size=32
augmentation=eta

dequant=flow
dequant_steps=4
dequant_context=32

num_scales=3
num_steps=6
coupling_network=transformer
coupling_blocks=12
coupling_channels=96
coupling_depth=1            # only for convolutions and densenets
coupling_gated_conv=False   # only for densenets
coupling_dropout=0.0
coupling_mixtures=4

conditional_channels=""
lowres_upsampler_channels="32 64"
lowres_encoder_channels=32
lowres_encoder_blocks=1
lowres_encoder_depth=3
compression_ratio="0.5"

latent_size=0
vae_hidden_units=""
base_distribution="c"

