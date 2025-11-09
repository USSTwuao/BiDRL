import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Deep reinforcement learning for solving the multi-objective mixed fleet vehicle routing problem")
    
    #data
    parser.add_argument('--problem', default='mfvrp', help="The problem to solve")
    parser.add_argument('--mfvrp_size', type=int, default=100, help="The size of the problem graph")
    parser.add_argument('--num_station',type=int, default=10, help="Number of station in instances: {3,5,7,9,11}")
    parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
    parser.add_argument('--epoch_size', type=int, default=1280000, help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance') 
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation') 
    parser.add_argument('--obj', default=[ 'min-cos'])

    #model
    parser.add_argument('--model', default='attention', help="Model, 'attention' ")
    parser.add_argument('--emb_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the Parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")

    #training
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--lr_model', type=float, default=0.0001, help="Set the learning rate for the actor network")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--lr_decay', type=float, default=0.995, help='Learning rate decay per epoch')
    parser.add_argument('--n_epochs', type=int, default=100, help='The number of epochs to train')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default=3.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)') 
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay (default 0.8)') 
    parser.add_argument('--baseline', default='rollout',  
                        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline') 
    parser.add_argument('--bl_warmup_epochs', type=int, default=1,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                             'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.') 
    parser.add_argument('--eval_batch_size', type=int, default=512,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--data_distribution', type=str, default=None,
                        help='Data distribution to use during training, defaults and options depend on problem.')
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    
    #Miscellaneous
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--load_path', help='Path to load model Parameters and optimizer state from')
    parser.add_argument('--no_progress_bar', action='store_false', help='Disable progress bar')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('-wd', '--weight_dir', metavar = 'MD', type = str, default = './Weights/', help = 'model weight save dir')
    

    
    opts = parser.parse_args(args)
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}_{}".format(opts.problem, opts.mfvrp_size,opts.num_station),
        opts.run_name
    )
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    return opts