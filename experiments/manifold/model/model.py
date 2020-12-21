from model.linear_flow import LinearFlow
from model.nonlinear_flow import NonlinearFlow
from model.pool_flow import PoolFlow


def add_model_args(parser):

    # Model choice
    parser.add_argument('--linear', type=eval, default=True)
    parser.add_argument('--stochastic_elbo', type=eval, default=False)
    parser.add_argument('--pooling', type=str, default='none', choices={'none', 'max'})

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
    if args.linear:
        if args.stochastic_elbo:
            return 'flow_and_linear_stochastic_vae'
        else:
            return 'flow_and_linear_analytic_vae'

    else:
        if args.pooling == "max":
            return 'pool_flow_vae'
        else:
            return 'flow_and_nonlinear_vae'


def get_model(args, data_shape, cond_shape=None):

    if args.linear:
        model = LinearFlow(data_shape=data_shape,
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
        if args.pooling == "max":
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
            model = NonlinearFlow(data_shape=data_shape,
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

    return model
