cd /export/scratch/robert/survae_flows

# Load defaults for all experiments
#source /export/scratch/robert/survae_flows/experiments/manifold/scripts/experiment_config.sh
seed=101
compression=vae
epochs=2
optimizer=adamax
max_grad_norm=1.0
warmup=100
exponential_lr=True
early_stop=10
eval_every=5
annealing_schedule=0

dataset=mnist
batch_size=32
device=cuda

trainable_sigma=True
latent_size=192
vae_hiden_units="384"
vae_activation=relu
base_distributions="n"

dequant=flow
dequant_steps=1
dequant_context=5

num_scales=2
num_steps=1
#coupling_network=transformer
coupling_blocks=1
coupling_channels=8
coupling_depth=1
coupling_growth=4
coupling_dropout=0.2
coupling_mixtures=4


for linear in False True
do

    for stochastic_elbo in True #False
    do
        if [[ ${stochastic_elbo} == 'False' && ${linear} == 'False' ]]; then
            continue
        fi

        for coupling_network in  densenet transformer conv
        do
            


            printf "\n---------- Running: ${compression}, linear=${linear}, stochastic=${stochastic_elbo}, coupling=${coupling_network} (${base_distributions}) ----------\n"
        
            python experiments/manifold/train.py \
                   --device               ${device} \
                   --seed                 ${seed} \
                   --dataset              ${dataset} \
                   \
                   --compression          ${compression} \
                   --linear               ${linear} \
                   --stochastic_elbo      ${stochastic_elbo} \
                   \
                   --batch_size           ${batch_size} \
                   --optimizer            ${optimizer} \
                   --epochs               ${epochs} \
                   --warmup               ${warmup} \
                   --exponential_lr       ${exponential_lr} \
                   --max_grad_norm        ${max_grad_norm} \
                   --annealing_schedule   ${annealing_schedule} \
                   --early_stop           ${early_stop} \
                   --eval_every           ${eval_every} \
                   \
                   --vae_activation       ${vae_activation} \
                   --vae_hidden_units     ${vae_hiden_units} \
                   --base_distributions   ${base_distributions} \
                   --trainable_sigma      ${trainable_sigma} \
                   --latent_size          ${latent_size} \
                   \
                   --num_scales           ${num_scales} \
                   --num_steps            ${num_steps} \
                   \
                   --dequant              ${dequant} \
                   --dequant_steps        ${dequant_steps} \
                   --dequant_context      ${dequant_context} \
                   \
                   --coupling_network     ${coupling_network} \
                   --coupling_blocks      ${coupling_blocks} \
                   --coupling_channels    ${coupling_channels} \
                   --coupling_depth       ${coupling_depth} \
                   --coupling_growth      ${coupling_growth} \
                   --coupling_dropout     ${coupling_dropout} \
                   --coupling_mixtures    ${coupling_mixtures} \
                   --name                 debug \
                ;
        done
    done
done


printf "\n\nJob complete!\n\n"
