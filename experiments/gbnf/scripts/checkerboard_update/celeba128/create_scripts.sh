source ~/super_resolution/survae_flows/experiments/gbnf/scripts/checkerboard_update/celeba128_config.sh
DATE=$(date '+%Y_%m_%d')

# Set batch size depending GPU config
TEMPLATE="../sr_template_2_v100.sh"
### TEMPLATE="../sr_template_1_v100.sh"
BATCHSIZE=32

SRX=8
TRANSFORMER="${coupling_blocks}b${coupling_channels}c${coupling_mixtures}m"
WANDB=True

for FLOW in slice; do
    for SEED in 101 102 103; do
        FNAME=${dataset}_${num_bits}bits_sr${SRX}x_${FLOW}_scales${num_scales}_steps${num_steps}_transformer${TRANSFORMER}_seed${SEED}.sh;
        cp ${TEMPLATE} ${FNAME};
        sed -i -e "s/DATASET/${dataset}/g" ${FNAME};
        sed -i -e "s/NUMBITS/${num_bits}/g" ${FNAME};
        sed -i -e "s/FLOW/${FLOW}/g" ${FNAME};
        sed -i -e "s/SCALES/${num_scales}/g" ${FNAME};
        sed -i -e "s/STEPS/${num_steps}/g" ${FNAME};
        sed -i -e "s/TRANSFORMER/${TRANSFORMER}/g" ${FNAME};
        sed -i -e "s/SEED/${SEED}/g" ${FNAME};
        sed -i -e "s/SRX/${SRX}/g" ${FNAME};
        sed -i -e "s/BATCHSIZE/${BATCHSIZE}/g" ${FNAME};
        sed -i -e "s/DATE/${DATE}/g" ${FNAME};
        sed -i -e "s/WANDB/${WANDB}/g" ${FNAME};
    done;
done;

