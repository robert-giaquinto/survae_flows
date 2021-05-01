#!/bin/bash -l        
#SBATCH --time=24:00:00
#SBATCH --ntasks=6
#SBATCH --mem=128g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=smit7982@umn.edu 
#SBATCH -p v100
#SBATCH --gres=gpu:v100:2
#SBATCH -o DATE_DATASET_NUMBITSbits_srSRXx_FLOW_scalesSCALES_stepsSTEPS_transformerTRANSFORMER_seedSEED.log
#SBATCH --job-name=DATE_DATASET_NUMBITSbits_srSRXx_FLOW_scalesSCALES_stepsSTEPS_transformerTRANSFORMER_seedSEED
module unload python
module load python/3.7.1_anaconda
module load python3/3.7.1_anaconda

cd ~/super_resolution/survae_flows

# activate virtual environment
source ~/super_resolution/survae_flows/venv/bin/activate

# Load defaults for DATASET experiments
source ~/super_resolution/survae_flows/experiments/gbnf/scripts/checkerboard_update/DATASET_config.sh

printf "\nTraining Flows with base arguments:\n"
cat ~/super_resolution/survae_flows/experiments/gbnf/scripts/checkerboard_update/DATASET_config.sh

device=cuda
amp=True
parallel=dp
num_workers=4

seed=SEED
batch_size=BATCHSIZE
log_wandb=WANDB
flow=FLOW
sr_scale_factor=SRX


# --resume                    "2021" \
# --name                      "2021_cont1" \

time python experiments/gbnf/train.py \
    --device                    ${device} \
    --parallel                  ${parallel} \
    --amp                       ${amp} \
    --seed                      ${seed} \
    --log_wandb                 ${log_wandb} \
    \
    --dataset                   ${dataset} \
    --num_bits                  ${num_bits} \
    --augmentation              ${augmentation} \
    --num_workers               ${num_workers} \
    \
    --batch_size                ${batch_size} \
    --optimizer                 ${optimizer} \
    --epochs                    ${epochs} \
    --warmup                    ${warmup} \
    --exponential_lr            ${exponential_lr} \
    --max_grad_norm             ${max_grad_norm} \
    --early_stop                ${early_stop} \
    --eval_every                ${eval_every} \
    --check_every               ${check_every} \
    \
    --flow                      ${flow} \
    --super_resolution          \
    --sr_scale_factor           ${sr_scale_factor} \
    --compression_ratio         ${compression_ratio} \
    \
    --num_scales                ${num_scales} \
    --num_steps                 ${num_steps} \
    --checkerboard_scales       ${checkerboard_scales} \
    \
    --coupling_network          ${coupling_network} \
    --coupling_blocks           ${coupling_blocks} \
    --coupling_channels         ${coupling_channels} \
    --coupling_dropout          ${coupling_dropout} \
    --coupling_mixtures         ${coupling_mixtures} \
    --actnorm                   ${actnorm} \
    \
    --conditional_channels      ${conditional_channels} \
    --lowres_upsampler_channels ${lowres_upsampler_channels} \
    --lowres_encoder_channels   ${lowres_encoder_channels} \
    --lowres_encoder_blocks     ${lowres_encoder_blocks} \
    --lowres_encoder_depth      ${lowres_encoder_depth} \
    \
    --augment_size              ${augment_size} \
    --augment_steps             ${augment_steps} \
    --augment_context           ${augment_context} \
    --augment_blocks            ${augment_blocks} \
    \
    --dequant                   ${dequant} \
    --dequant_steps             ${dequant_steps} \
    --dequant_context           ${dequant_context} \
    --dequant_blocks            ${dequant_blocks} \
    ;

echo "Job complete"
