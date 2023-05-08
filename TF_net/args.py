import argparse
def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    # Main Params
    parser.add_argument("--data",
                        type=str,
                        default="rbc_data.pt")
    parser.add_argument("--min_mse",
                        type=int,
                        default=float('inf'))
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
    parser.add_argument("--dropout_rate",
                        type=float,
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
    parser.add_argument("--wt_decay",
                            help="weight decay",
                            default=4e-4,
                            type=float) 
    parser.add_argument("--not_use_test_mode",
                        action="store_true",
                        default=False,
                        help="test with same config as train mode, i.e. for rbc_data, at test time too divide the data into 7 parts.")
    parser.add_argument("--only_val", 
                        action='store_true', 
                        help="only test on validation",
                        default=False)
    parser.add_argument("--desc",
                        type=str,
                        default="")
    parser.add_argument("--inv_transform", 
                        action='store_true', 
                        help="instead of simply multiplying by beta, use inv_transform for testing",
                        default=False)
    parser.add_argument("--num_workers",
                        type=int,
                        default=8)
    parser.add_argument("--mixed_indx", 
                        action='store_true', 
                        help="mix val and test indexes. Currently, only supported for data* datasets",
                        default=False)
    parser.add_argument("--mixed_indx_no_overlap", 
                        action='store_true', 
                        help="mix val and test indexes but without any overlap. Currently, only supported for data* datasets",
                        default=False)
    parser.add_argument("--version",
                        type=str,
                        help="dataset version to used, this version is used for testing, while training is done on some other version of dataset",
                        nargs="?",
                        const="",
                        default="")
    parser.add_argument("--rescale_loss",
                        help="Rescale loss by the output_length",
                        action="store_true",
                        default=False)
    parser.add_argument("--norm_loss",
                        help="simply divide the loss by the output_length",
                        action="store_true",
                        default=False)

    # warmup
    parser.add_argument("--warm_up_epochs",
                        type=int,
                        help="Number of warmup epochs, number of epochs will be increased by warmupepochs",
                        default=0)
    parser.add_argument("--warm_up_min_lr",
                        type=float,
                        help="lr to start with the warmup",
                        default=1e-6)
    
    # optimizer args
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.001)
    parser.add_argument("--gamma",
                        type=float,
                        default=0.9)
    parser.add_argument("--sch", 
                        help="type of scheduler", 
                        type = str, default='')
    parser.add_argument("--tmax",
                        type=int,
                        default=100)
    
    
    # mide lyapunov param
    parser.add_argument("--mide",
                        type=float,
                        help="Mide Value, if None, then learning is used for m.",
                        default=None)
    parser.add_argument("--m_init",
                        help="If mide is None, thus we learn m, then m_init is used as initialization",
                        type=float,
                        default=0.09)
    
    #slope value, reduce these value if you get a nan gradient in Lyapunov.
    parser.add_argument("--slope",
                        type=int,
                        default=300) # 300
    parser.add_argument("--slope_init",
                        help="if slope is None, i.e. slope is learnt, init it with this value",
                        type=int,
                        default=300) # 300
    
    # Use time in learning
    parser.add_argument("--use_time", 
                        help="use time as well for (mse - mide) transformation", 
                        action='store_true', default=False)
    parser.add_argument("--time_factor",
                        type=float,
                        default=0.01)
    
    # Lyapunov fine-grained params
    parser.add_argument("--barrier",
                        type=float,
                        default=1e-3)
    parser.add_argument("--d_ids",
                        nargs='+',
                        type=int,
                        help="gpu id list to use",
                        default=[2])
    parser.add_argument("--bnorm", 
                        action='store_true', 
                        help="whether to use relu count only or entire batch len for normalization in Lyapunov",
                        default=False)
    parser.add_argument("--no_weight", 
                        action='store_true', 
                        help="No weight scheduler",
                        default=False)
    parser.add_argument("--teacher_forcing", 
                        action='store_true', 
                        help="use teacher forcing during training",
                        default=False)
    
    # Output length controls
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
    
    # Using noise
    parser.add_argument("--noise",
                        type=float,
                        help="sigma = 0.01*noise",
                        default = 0.0)
    parser.add_argument("--dnsn",
                        help="'do not scale noise', multiply gaussian with the element",
                        default = False,
                        action="store_true")
    
    # Additional encoder-decoder
    parser.add_argument("--addon_enc",
                            help="number of additonal layers to encoder",
                            type=int,
                            default = 0)
    parser.add_argument("--addon_dec",
                            help="number of additonal layers to decoder",
                            type=int,
                            default = 0)
    
    # Masking
    parser.add_argument("--mask",
                        action="store_true",
                        default=False,
                        help="use masking")
    parser.add_argument("--mtype",
                        help="masking type",
                        choices=["random", "opt"],
                        default="random")
    parser.add_argument("--mlower",
                        type=float,
                        help="lower bound of mask_ratio",
                        default = 80.0)
    parser.add_argument("--mupper",
                        type=float,
                        help="upper bound of mask_ratio",
                        default = 100.0)
    parser.add_argument("--mstart",
                        type=int,
                        help="at epochs <= mstart, mask_ratio == mlower",
                        default = 15)
    parser.add_argument("--mend",
                        type=int,
                        help="at epochs >= mend, mask_ratio == upper",
                        default = 50)
    parser.add_argument("--mtile",
                        type=int,
                        help="tile_size",
                        default = 16)
    parser.add_argument("--mwarmup",
                        type=int,
                        help="use full 100% masking until warmup to avoid learning bad interpolation functions",
                        default = 0)
    parser.add_argument("--mfloss",
                        action="store_true",
                        help="use loss on both predicted and unpredicted",
                        default = False)
    parser.add_argument("--mcurr",
                        action="store_true",
                        help="use masking for curriculum leanring",
                        default = False)
    
    # positional embedding
    parser.add_argument("--pos_emb_dim",
                        default=0,
                        type=int,
                        help="use positional embedding for time, with provided dim")
    
    # truncate loss idea
    parser.add_argument("--trunc",
                        type=int,
                        help="the output length at which a 'max_loss' param is recorded and all the future values higher that that are ignored.",
                        default=float('inf'))
    parser.add_argument("--trunc_factor",
                        type=float,
                        help="factor mult to trunc thresh",
                        default=0.7)
    
    # loss decay
    parser.add_argument("--beta",
                        type=float,
                        help="the output length at which a 'max_loss' param is recorded and all the future values higher that that are ignored.",
                        default=1)
    
    # gmm params
    parser.add_argument("--gmm_comp",
                        type=int,
                        help="Number of components used for GMM, GMM only used if gmm_comp > 0",
                        default=0)

    return parser.parse_args(args=args)