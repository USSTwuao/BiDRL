import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import json
import pprint as pp
import torch, gc
gc.collect()
import torch.optim as optim
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselies import NoBaseline, ExponentialBaseline,  RolloutBaseline, WarmupBaseline
from nets.Attention_model import AttentionModel
from problems.problem_mfvrp import MFVRP
from utils.functions import torch_load_cpu
import warnings
torch.cuda.empty_cache()
def run(opts):

    tb_logger = None
    torch.manual_seed(6) 
    os.makedirs(opts.save_dir) 
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:json.dump(vars(opts), f, indent=True)
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")
    problem = MFVRP()

    load_data = {}
    load_path = opts.load_path

    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)
    else:
        print('  [*] No load path specified, skipping loading data.')

    model_class = {
        'attention': AttentionModel}.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)


    model = model_class(
        opts.emb_dim,
        opts.hidden_dim,
        problem,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=False,
    ).to(opts.device)
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, opts)

    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)
    
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}])
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    val_dataset = problem.make_dataset(
        size=opts.mfvrp_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

    if opts.eval_only:
        validate(model, val_dataset, opts)
    
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            gc.collect()
            torch.cuda.empty_cache()
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                tb_logger,
                opts
            )


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run(get_options())
