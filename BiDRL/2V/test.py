import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
torch.cuda.memory_reserved(device=torch.device('cuda'))
import pickle
import time 
time.sleep(2)
torch.cuda.empty_cache()
from nets.Attention_model import AttentionModel
from options import get_options
from problems.problem_mfvrp import MFVRP
from nets.Attention_model import set_decode_type

torch.cuda.empty_cache()

current_dir = os.path.dirname(os.path.abspath(__file__))

opts = get_options()
problem = MFVRP()
opts.device = torch.device("cuda:0" )
model = AttentionModel(
    opts.emb_dim,
    opts.hidden_dim,
    problem,
    mask_inner=True,
    mask_logits=True,
    normalization=opts.normalization,
    tanh_clipping=opts.tanh_clipping,
    checkpoint_encoder=False,
).to(opts.device)

model_path = os.path.join(current_dir, 'Weights', 'model.pt')   

if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    set_decode_type(model, "sampling")
else:
    print(f"Model file not found at {model_path}")

data_path = os.path.join(current_dir,  "data", "mfvrp", "data.pkl")
if os.path.exists(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
else:
    print(f"Data file not found at {data_path}")

data = torch.tensor(data)
data = data.to(opts.device)
expanded_data = data.repeat(512, 1, 1)

start_time = time.time()
for i in range(25):
    print(f"Run {i + 1}:")

    with torch.no_grad():
        cost, _, _, node_selected_list, veh_select_list = model(expanded_data)
        min_value, min_index = torch.min(cost, dim=0)

    print(f"最小值: {min_value.item()}")
    print("节点顺序", node_selected_list[min_index])
    print("车辆顺序", veh_select_list[min_index])

end_time = time.time()


total_time = end_time - start_time
print(f"Total time taken for 25 runs: {total_time:.2f} seconds")






