from model.linear_flow import NDPLinearFlow
from model.nonlinear_flow import NDPNonlinearFlow
from model.pool_flow import PoolFlow
from model.vae_flow import VAEFlow


def add_model_args(parser):

    # Model choice
    parser.add_argument('--linear', type=eval, default=True)
    parser.add_argument('--stochastic_elbo', type=eval, default=False)
    parser.add_argument('--pooling', type=str, default='none', choices={'none', 'max'})
    parser.add_argument('--gaussian_mid', type=eval, default=False)

    # VAE params
    parser.add_argument('--latent_size', type=int, default=196)
    parser.add_argument('--trainable_sigma', type=eval, default=True)
    parser.add_argument('--sigma_init', type=float, default=1.0)
    parser.add_argument('--vae_hidden_units', nargs="+", type=int, default=[512, 256])
    parser.add_argument('--vae_activation', type=str, default='none')

    # Flow params
    parser.add_argument('--num_scales', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--actnorm', type=eval, default=True)

    # Dequant params
    parser.add_argument('--dequant', type=str, default='uniform', choices={'uniform', 'flow'})
    parser.add_argument('--dequant_steps', type=int, default=4)
    parser.add_argument('--dequant_context', type=int, default=32)

    # Net params
    parser.add_argument('--densenet_blocks', type=int, default=1)
    parser.add_argument('--densenet_channels', type=int, default=64)
    parser.add_argument('--densenet_depth', type=int, default=2)
    parser.add_argument('--densenet_growth', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gated_conv', type=eval, default=True)


def get_model_id(args):
    # Todo: include other key model parameters as part of id
    arch = f"scales{args.num_scales}_steps{args.num_steps}"
    if args.linear:
        if args.stochastic_elbo:
            model_id = f'NDP_Linear_Stochastic_Flow_latent{args.latent_size}'
        else:
            model_id = f'NDP_Linear_Analytical_Flow_latent{args.latent_size}'

    else:
        if args.gaussian_mid:
            model_id = f"NDP_Regularized_VAE_{args.vae_activation}_{'_'.join([str(elt) for elt in args.vae_hidden_units])}_Flow_latent{args.latent_size}"
        elif args.pooling == "none":
            model_id = f"NDP_VAE_{args.vae_activation}_{'_'.join([str(elt) for elt in args.vae_hidden_units])}_Flow_latent{args.latent_size}"
        elif args.pooling == "max":
            model_id = 'Max_Pool_Flow'
        else:
            raise ValueError(f"No model defined for {args.pooling} pooling")


    return model_id + "_" + arch


def get_model(args, data_shape, cond_shape=None):

    if args.linear:
        model = NDPLinearFlow(data_shape=data_shape,
            num_bits=args.num_bits,
            num_scales=args.num_scales,
            num_steps=args.num_steps,
            actnorm=args.actnorm,
            pooling=args.pooling,
            dequant=args.dequant,
            dequant_steps=args.dequant_steps,
            dequant_context=args.dequant_context,
            densenet_blocks=args.densenet_blocks,
            densenet_channels=args.densenet_channels,
            densenet_depth=args.densenet_depth,
            densenet_growth=args.densenet_growth,
            dropout=args.dropout,
            gated_conv=args.gated_conv,
            latent_size=args.latent_size,
            trainable_sigma=args.trainable_sigma,
            sigma_init=args.sigma_init,
            stochastic_elbo=args.stochastic_elbo)

    else:
        
        if args.gaussian_mid:

            # NDP Flow but the input to the VAE is Gaussian
            model = NDPNonlinearFlow(data_shape=data_shape,
                num_bits=args.num_bits,
                num_scales=args.num_scales,
                num_steps=args.num_steps,
                actnorm=args.actnorm,
                pooling=args.pooling,
                dequant=args.dequant,
                dequant_steps=args.dequant_steps,
                dequant_context=args.dequant_context,
                densenet_blocks=args.densenet_blocks,
                densenet_channels=args.densenet_channels,
                densenet_depth=args.densenet_depth,
                densenet_growth=args.densenet_growth,
                dropout=args.dropout,
                gated_conv=args.gated_conv,
                vae_hidden_units=args.vae_hidden_units,
                latent_size=args.latent_size,
                vae_activation=args.vae_activation)

            
        elif args.pooling == "none":

            # NDP Flow but the input to the VAE doesn't need to be Gaussian
            model = VAEFlow(data_shape=data_shape,
                num_bits=args.num_bits,
                num_scales=args.num_scales,
                num_steps=args.num_steps,
                actnorm=args.actnorm,
                pooling=args.pooling,
                dequant=args.dequant,
                dequant_steps=args.dequant_steps,
                dequant_context=args.dequant_context,
                densenet_blocks=args.densenet_blocks,
                densenet_channels=args.densenet_channels,
                densenet_depth=args.densenet_depth,
                densenet_growth=args.densenet_growth,
                dropout=args.dropout,
                gated_conv=args.gated_conv,
                vae_hidden_units=args.vae_hidden_units,
                latent_size=args.latent_size,
                vae_activation=args.vae_activation)

        elif args.pooling == "max":

            # Max pooling surjective flow
            model = PoolFlow(data_shape=data_shape,
                num_bits=args.num_bits,
                num_scales=args.num_scales,
                num_steps=args.num_steps,
                actnorm=args.actnorm,
                pooling=args.pooling,
                dequant=args.dequant,
                dequant_steps=args.dequant_steps,
                dequant_context=args.dequant_context,
                densenet_blocks=args.densenet_blocks,
                densenet_channels=args.densenet_channels,
                densenet_depth=args.densenet_depth,
                densenet_growth=args.densenet_growth,
                dropout=args.dropout,
                gated_conv=args.gated_conv)
            
        else:
            raise ValueError(f"No model defined for {args.pooling} pooling")

    return model
