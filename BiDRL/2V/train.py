import os
import time
from tqdm import tqdm
import torch
import math
from nets.Attention_model import set_decode_type
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import matplotlib.pyplot as plt
from problems.problem_mfvrp import MFVRP
from utils.functions import move_to
import torch.nn as nn
def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def rollout(model, dataset, opts):
    set_decode_type(model, "greedy")
    model.eval()
    all_costs = []

    def eval_model_bat(bat):
        with torch.no_grad(): 
            bat = bat.to(opts.device)
            cost, _, _ ,_,_= model(bat)
        return cost
    
    for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size),disable=opts.no_progress_bar):
        cost = eval_model_bat(bat)
        all_costs.append(cost)
    final_costs = torch.cat(all_costs, dim=0)
    return final_costs 

def validate(model, dataset, opts): 
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean(dim=0)
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    return avg_cost



def clip_grad_norms(param_groups, max_norm=math.inf):
    """ 
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped



def plot_grad_flow(model): 

    named_parameters = model.named_parameters() 
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())

    fig = plt.figure(figsize=(8,6))
    plt.plot(ave_grads, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))

    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()
    lr_scheduler.step(epoch) 



    training_dataset = baseline.wrap_dataset(MFVRP.make_dataset(size=opts.mfvrp_size, num_samples=opts.epoch_size,distribution=opts.data_distribution))  
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0) 
    task = "MFVRP30"
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    avg_reward = validate(model, val_dataset, opts)
    print(f"第{epoch}次训练,测试集cost均值为{avg_reward}")

    baseline.epoch_callback(model, epoch)
    file_path = os.path.join(opts.weight_dir, '%s_epoch%s.pt' % (task, epoch))

    if not os.path.exists(opts.weight_dir):
        os.makedirs(opts.weight_dir)

    torch.save(model.state_dict(), file_path)


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):  
    x, bl_val = baseline.unwrap_batch(batch) 
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
    cost, ll ,ll_veh ,_,_= model(x)

    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
    reinforce_loss = ((cost - bl_val)* (ll + ll_veh)).mean()
    loss = reinforce_loss
    loss.requires_grad_(True)
    optimizer.zero_grad()
    loss.backward()

    '''
    for name, param in model.named_parameters():
        print(f"{name} requires grad: {param.requires_grad}")
        if param.grad is not None:
            print(f"Gradient of {name}: shape {param.grad.shape}, norm {param.grad.norm()}")
        else:
            print(f"Gradient of {name}: None") '''

    nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0, norm_type = 2)
    
    optimizer.step()

   