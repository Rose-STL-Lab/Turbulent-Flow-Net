import argparse
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_mse",
                        type=int,
                        default=1)
    parser.add_argument("--seed",
                        type=int,
                        default=53)
    parser.add_argument("--time_range",
                        type=int,
                        default=6)
    parser.add_argument("--output_length",
                        type=int,
                        default=4)
    parser.add_argument("--input_length",
                        type=int,
                        default=26)
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.001)
    parser.add_argument("--dropout_rate",
                        type=int,
                        default=0)
    parser.add_argument("--kernel_size",
                        type=int,
                        default=3)
    parser.add_argument("--batch_size",
                        type=int,
                        default=32)
    parser.add_argument("--coef",
                        type=float,
                        default=0)
    parser.add_argument("--coef2",
                        type=float,
                        default=1)
    parser.add_argument("--step_size",
                        type=int,
                        default=1)
    parser.add_argument("--path",
                        type=str,
                        default="./results/")
    parser.add_argument("--epoch",
                        type=int,
                        default=100)
    parser.add_argument("--mide",
                        type=float,
                        default=None)
    parser.add_argument("--m_init",
                        type=float,
                        default=0.09)
    parser.add_argument("--slope",
                        type=int,
                        default=300) # 300
    parser.add_argument("--slope_init",
                        help="if slope is None, i.e. slope is learnt, init it with this value",
                        type=int,
                        default=300) # 300  # Reduce this value if you get a nan gradient
    parser.add_argument("--barrier",
                        type=float,
                        default=1e-3)
    parser.add_argument("--d_ids",
                        nargs='+',
                        type=int,
                        default=[2])
    parser.add_argument("--bnorm", 
                        action='store_true', 
                        default=False)
    parser.add_argument("--no_weight", 
                        action='store_true', 
                        default=False)
    parser.add_argument("--only_val", 
                        action='store_true', 
                        default=False)
    parser.add_argument("--use_time", 
                        help="use time as well for (mse - mide) transformation", 
                        action='store_true', default=False)
    parser.add_argument("--time_factor",
                        type=float,
                        default=0.01)
    parser.add_argument("--desc",
                        type=str,
                        default="")
    parser.add_argument("--data",
                        type=str,
                        default="rbc_data.pt")
    parser.add_argument("--outln_steps",
                        type=int,
                        default=0)
    parser.add_argument("--outln_init",
                        type=int,
                        help="init if warmuping",
                        default = 4) 
    parser.add_argument("--outln_stride",
                        type=int,
                        help="stride to take, if None, then stride = args.output_length - args.outln_init, that is one jump",
                        default = None)  
    parser.add_argument("--noise",
                        type=int,
                        help="sigma = 0.01*noise",
                        default = 0.0)
    parser.add_argument("--dnsn",
                        help="'do not scale noise', multiply gaussian with the element",
                        default = False,
                        action="store_true")
    parser.add_argument("--wt_decay",
                            help="weight decay",
                            default=4e-4,
                            type=float) 
    parser.add_argument("--addon_enc",
                            help="number of additonal layers to encoder",
                            type=int,
                            default = 0)
    parser.add_argument("--addon_dec",
                            help="number of additonal layers to decoder",
                            type=int,
                            default = 0)
    parser.add_argument("--not_use_test_mode",
                        action="store_true",
                        default=False,
                        help="test with same config as train mode")
    return parser.parse_args()
