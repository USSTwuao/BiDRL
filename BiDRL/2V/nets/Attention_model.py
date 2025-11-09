import torch
import torch.nn as nn
import math
from torch.nn import DataParallel
from typing import NamedTuple
from nets.GAT_encoder import GraphAttentionEncoder
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from problems.state_mfvrp import StateMFVRP


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type) 


class AttentionModelFixed(NamedTuple):  
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """

    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor


    

class AttentionModel(nn.Module):

    def __init__(self,
                 emb_dim,
                 hidden_dim,
                 problem,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 ): 
        super(AttentionModel, self).__init__() 

        self.checkpoint_encoder = checkpoint_encoder
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.decode_type = None
        self.temp = 1.0
        self.is_mfvrp = problem.NAME == 'mfvrp'
        self.feed_forward_hidden = 512
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.problem = problem
        self.n_heads = n_heads
        step_context_dim = emb_dim+3 
        self.batch_size = 512
        num_veh = 2
        node_dim = 6
        edge_dim = 1
        veh_dim = 5
        self.capacity = torch.tensor([4.6,3.2]).to('cuda:0')
        self.consume =torch.tensor([0.0426,0.1436]).to('cuda:0')
        self.energy = torch.tensor([120.,106.]).to('cuda:0')
        self.weight = torch.tensor([3.4,4.1])
        self.sequence = []

        self.FF_veh = nn.Sequential(nn.Linear(veh_dim, self.emb_dim),
                nn.Linear(self.emb_dim, self.feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(self.feed_forward_hidden, 128)
            )



        self.embedder = GraphAttentionEncoder( 
        edge_dim = edge_dim,
        batch_size = 512,
        dropout = 0,
        alpha = 0.2,
        node_dim = node_dim,
        n_heads = n_heads,
        emb_dim = emb_dim,
        normalization=normalization,
        feed_forward_hidden=self.feed_forward_hidden,
        )

        self.project_node_embeddings = nn.Linear(emb_dim, 3 * emb_dim, bias=False) 
        self.project_fixed_context = nn.Linear(emb_dim, emb_dim, bias=False)  
        self.project_step_context = nn.Linear(step_context_dim, emb_dim, bias=False)  
        self.veh_select_context  = nn.Linear(emb_dim*2,1,bias=False)  
        assert emb_dim % n_heads == 0
        self.project_out = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.bilinear_pooling = BilinearPooling(128, 128)


    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type 
        if temp is not None: 
            self.temp = temp  

    
    def _precompute(self, node_embed):  
        graph_embed = node_embed.mean(dim=1) 
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(node_embed[:, None, :, :]).chunk(3,dim=-1)  

        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed), 
            self._make_heads(glimpse_val_fixed), 
            logit_key_fixed.contiguous()  
        )
        return AttentionModelFixed(node_embed, fixed_context, *fixed_attention_node_data)
    

    def _inner(self, input, node_embed): 
        state = self.problem.make_state(input)   

        node_coord = state.coords 
        dis_matrix=(node_coord[:, :, None, :] - node_coord[:, None, :, :]).norm(p=2, dim=-1)  
    
        device = state.coords.device
        current_node = state.pre_a
        batch_size= current_node.size(0)
        start_cost = torch.tensor([80,73]).to(device)
        veh_select_list=torch.zeros(batch_size,1).to(device)
        node_selected_list = torch.zeros(batch_size,1).to(device) 
        fixed = self._precompute(node_embed)

        over = state.not_all_finished()
        node_select_logp = []
        veh_select_logp = []
        while over:
            current_node = state.pre_a
            if torch.all(current_node == 0): 
                state.i = 0
                veh , log_veh_select= self.select_veh(state ,node_embed) 
                veh_select_list = torch.cat((veh_select_list,veh),dim = 1)
                state.surplus_capacity = self.capacity[veh]
                state.surplus_energy = self.energy[veh]
                state.cur_veh = veh
                state.cur_time = torch.zeros(batch_size,dtype=torch.int)
                selected,node_prob,mask,cunzai,rows_with_all_false = self.select_node(state, node_embed ,fixed ,dis_matrix)
                if torch.all(selected == 0):
                    break
                else:
                    conditionA = (selected == 0)
                    maskA = conditionA
                    maskB = ~conditionA
                    state = state.update(selected ,veh)
                    
                    state.cur_Ecost[maskB] = state.cur_Ecost[maskB] + start_cost[veh][maskB]
                    state.cur_Ecost[maskA] = state.cur_Ecost[maskA]

            else:
                cur_veh = state.cur_veh
                selected,node_prob,mask,cunzai,rows_with_all_false = self.select_node(state, node_embed ,fixed ,dis_matrix)
                state = state.update(selected ,cur_veh)

            if cunzai:
                print("data",input[rows_with_all_false[0]])
                print("veh",veh_select_list[rows_with_all_false[0]])
                print("node",node_selected_list[rows_with_all_false[0]])
                raise ValueError("Mask contains at least one row with all False values.")
            
            node_prob = node_prob * mask.float()
            veh_select_logp.append(log_veh_select.unsqueeze(1)) 
            node_select_logp.append(node_prob.unsqueeze(1))  

            over = state.not_all_finished()

            node_selected_list = torch.cat((node_selected_list ,selected),dim = 1)

        veh_select_prob = torch.cat(veh_select_logp,dim=1)  
        node_select_prob = torch.cat(node_select_logp,dim=1)    
        veh_select_list = veh_select_list[:, 1:]
        ll = node_select_prob[torch.arange(node_select_prob.size(0)).unsqueeze(1).long(), torch.arange(node_select_prob.size(1)).unsqueeze(0).long(), node_selected_list[:, 1:].long()]
        ll_veh = veh_select_prob[torch.arange(veh_select_list.size(0)).unsqueeze(1).long(), torch.arange(veh_select_list.size(1)).unsqueeze(0).long(), veh_select_list.long()]
            
        ll = ll.sum(dim=1)
        ll_veh = ll_veh.sum(dim=1)
        return state, ll ,ll_veh,node_selected_list, veh_select_list



    def forward(self, input):
        if self.checkpoint_encoder:
            embeddings = checkpoint(self.embedder, input)  

        else:
            embeddings,_ = self.embedder(input)

        state, ll ,ll_veh,node_selected_list, veh_select_list = self._inner(input ,embeddings)
        cost = self.problem.get_costs(state)  
        cost = cost.sum(dim=1,keepdim=True)
        return cost ,ll ,ll_veh,node_selected_list, veh_select_list


    def select_veh(self, state, node_embed): 
        device = state.coords.device
        N = node_embed.shape[1]
        graph_embedhe = node_embed.sum(dim=1).to(device) 
        graph_embed = graph_embedhe.unsqueeze(1).repeat(1, 2, 1) 
        graph_embed = graph_embed.clamp(min=1e-6)  

        batch_size = self.batch_size
        
        cap = torch.tensor([4.6,3.2], device=device)
        energy = torch.tensor([120.,106.], device=device)
        start_cost = torch.tensor([80,73], device=device)
        consume = torch.tensor([0.0426,0.1436], device=device)
        speed = torch.tensor([1.1,1], device=device)
        weight = torch.tensor([3.4,4.1], device=device)
        prize = torch.tensor([7.2,1.3], device=device) 
        
        matrix = torch.stack((cap, energy, weight, consume,speed), dim=1).to(device)  
        veh_fea = matrix.unsqueeze(0).repeat(batch_size, 1, 1).to(device)  
        veh_context = self.FF_veh(veh_fea).to(device) 

        visited = state.visited_.to(device)
        mask = (visited == 0).to(device)

        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1) 
        masked_node_embed = node_embed.to(device) * mask.unsqueeze(-1) 
        masked_node_embed_sum = masked_node_embed.sum(dim=1)
        unvisited_node_context = masked_node_embed_sum  
        unvisited_node_context = unvisited_node_context.unsqueeze(1).repeat(1, 2, 1) 
        unvisited_node_context = unvisited_node_context 
        graph_embed = graph_embed
        veh_context_with_node = torch.cat((unvisited_node_context/N,veh_context),dim=2)  



        veh_select_pro = self.veh_select_context(veh_context_with_node).squeeze(-1) 


        log_veh_select = F.log_softmax(veh_select_pro, dim=1)

        if self.decode_type == "greedy": 
            _, veh = log_veh_select.exp().max(1)
            veh = veh.unsqueeze(1)

        elif self.decode_type == "sampling":
            veh = log_veh_select.exp().multinomial(1) 

        return veh, log_veh_select
        


    def get_mask(self, state,dis_matrix):
        
        device = state.coords.device
        speed_veh = torch.tensor([1.1,1]).to(device)
        weight = torch.tensor([3.4,4.1]).to(device)

        if state.i == 0:
            visited = state.visited_.clone()
            num_station = state.num_station
            visited[:,-num_station :] = 1
            visited[:, 0] = 1
            full_one_rows = torch.all(visited == 1, dim=1)
            visited[full_one_rows, 0] = 0
            mask = (visited == 0) 
            state.visited_[:,-num_station :] = 0
            cunzai = False
            rows_with_all_false =[]
        else:
            visited = state.visited_
            mask_node = (visited == 0).to(device)  
            num_station = int(state.num_station)
            cur_veh = state.cur_veh
            conditionA = ((cur_veh < 1)).squeeze()
            mask_node[conditionA,-num_station:]= False
            batch_size , N = visited.size()
            used_capacity = state.used_capacity
            all_weight = used_capacity+weight[cur_veh]
            surplus_capacity = state.surplus_capacity
            surplus_energy = state.surplus_energy
            cur_time = state.cur_time
            cur_node = state.pre_a

            mask_cur_0 = torch.ones((batch_size, N), dtype=torch.bool).to(device)
            zero_indices = torch.where(cur_node == 0)[0] 
            mask_cur_0[zero_indices, 1:] = False

            demand = state.demand
            surplus_capacity_compare = surplus_capacity.expand_as(demand)
            surplus_energy_compare = surplus_energy.expand_as(demand)
        

            mask_cap = (demand<=surplus_capacity_compare).to(device)


            cur_node_dis_matrix = (dis_matrix[torch.arange(batch_size).unsqueeze(1),cur_node]).squeeze(1) #[batch_size,N]


            road1_energy_consumption = all_weight.repeat(1, N) * cur_node_dis_matrix * (self.consume[state.cur_veh]).repeat(1, N)
            mask_energy1 = (road1_energy_consumption <= surplus_energy_compare).to(device)

            num_station = int(state.num_station)
            dis_node_to_0orsta = torch.cat((dis_matrix[:, :, :1], dis_matrix[:, :, -num_station:]), dim=2)  #[b, N, 4]
            can_go_0orsta = torch.cat((mask_node[:, :1], mask_node[:, -num_station:]), dim=1)  #[b, 4]
            can_go_0orsta = can_go_0orsta.unsqueeze(1).repeat(1, N, 1)   #[batch_size,N,4]
            dis_node_to_0orsta[~can_go_0orsta] = 400
            dis_node_to_0orsta , _ = torch.min (dis_node_to_0orsta, dim=2)  #[batch_size, N]
            road2_energy_consumption = (all_weight + demand) * dis_node_to_0orsta * (self.consume[state.cur_veh]).repeat(1, N)
            lowest_energy = road1_energy_consumption + road2_energy_consumption   #[batch_size,N]
            mask_energy2 = (lowest_energy<=surplus_energy_compare).to(device)

            late_timewindow = state.last_time   #[batch_size,N]
            arrive_time = cur_time + cur_node_dis_matrix/speed_veh[state.cur_veh] 
        
            mask_time = (arrive_time<=late_timewindow).to(device) 

            mask_sta = torch.ones(batch_size, N, dtype=torch.bool).to(device)
            condition = ((cur_node > state.mfvrp_size)| (cur_node == 0)).squeeze()
            mask_sta[condition,-num_station:]= False

            mask_fuel = torch.ones(batch_size, N, dtype=torch.bool).to(device)
            cur_veh = state.cur_veh
            conditionA = ((cur_veh < 1)).squeeze()
            mask_fuel[conditionA,-num_station:]= False
            
            mask = mask_node & mask_cap & mask_energy1 & mask_energy2 & mask_time & mask_cur_0 & mask_sta & mask_fuel
            
            rows_with_all_false = torch.all(mask == False, dim=1).nonzero(as_tuple=True)[0]
            cunzai = False
            
        return mask,cunzai,rows_with_all_false


    def select_node(self, state, node_embed ,fixed ,dis_matrix):
        device = state.coords.device
        graph_embed = node_embed.mean(dim=1)
        context_embed = self._get_parallel_step_context(node_embed , graph_embed , state)  #[batch_size,128]
        
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)

        mask,cunzai,rows_with_all_false = self.get_mask(state,dis_matrix)
        mask = mask.to(device)

        node_prob = self.get_node_select_prob(context_embed ,glimpse_K, glimpse_V, logit_K, mask)
        assert (node_prob == node_prob).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = node_prob.exp().max(1)
            selected = selected.unsqueeze(1)

        elif self.decode_type == "sampling":
            selected = node_prob.exp().multinomial(1)

        else:
            assert False, "Unknown decode type"

        return selected,node_prob,mask,cunzai,rows_with_all_false

    def get_node_select_prob(self, query, glimpse_K, glimpse_V, logit_K, mask):
        batch_size, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads
    
        glimpse_Q = query.view(batch_size, self.n_heads, 1, key_size).permute(1, 0, 2, 3)
        assert not torch.isnan(glimpse_Q).any(), "NaN in glimpse_Q"
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        assert not torch.isnan(compatibility).any(), "NaN in compatibility before mask application"
    
        compatibility[~mask[None, :, None, :].repeat(8, 1, 1, 1)] = -1e9
        assert not torch.isnan(compatibility).any(), "NaN in compatibility after mask application"
    
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)
        assert not torch.isnan(heads).any(), "NaN in heads"

        glimpse = self.project_out(heads.permute(1, 2, 0, 3).contiguous().view(-1, 1, self.n_heads * val_size))
        assert not torch.isnan(glimpse).any(), "NaN in glimpse"
    
        final_Q = glimpse
        logit_K_ = logit_K.transpose(-2, -1).squeeze(-3)
        logits = (torch.matmul(final_Q, logit_K_) / math.sqrt(final_Q.size(-1))).squeeze(1)
        assert not torch.isnan(logits).any(), "NaN in logits before tanh_clipping"

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
            assert not torch.isnan(logits).any(), "NaN in logits after tanh_clipping"

        if self.mask_logits:  # True
            logits[~mask] = -1e9
            assert not torch.isnan(logits).any(), "NaN in logits after mask application"

        log_p = F.log_softmax(logits / self.temp, dim=-1)
        assert not torch.isnan(log_p).any(), "NaN in log_p"

        return log_p



    def _get_parallel_step_context(self, node_embed, graph_embed,state ): 
        device = state.coords.device
        batch_size = node_embed.size(0)
        cur_node = state.pre_a
        cur_node_emb = node_embed[torch.arange(batch_size),cur_node.squeeze(1)]  
        surplus_capacity = state.surplus_capacity 
        surplus_energy = state.surplus_energy  
        cur_time = (state.cur_time).view(batch_size,1)   

        cur_time = cur_time.to(device)


        context_embed = torch.cat((
            cur_node_emb,  #[batch_size,128]
            surplus_capacity, #[batch_size,1]
            surplus_energy, #[batch_size,1]
            cur_time #[batch_size,1]
        ),dim = 1)
        context_embed = context_embed.float()

        return graph_embed + self.project_step_context(context_embed)  
    


    def _get_attention_node_data(self, fixed):  
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key


    def _make_heads(self, v): 

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)  #[batch_size, 1 ,N, n_heads , 128/8=16]
                .squeeze(1)    #[batch_size,  N, n_heads , 128/8=16]
                .permute(2, 0, 1, 3)  # (n_heads, batch_size, N, 16)
        ) 
    

class BilinearPooling(nn.Module):
    def __init__(self, node_embed_dim=128, veh_fea_dim=128, output_dim=256):
        super(BilinearPooling, self).__init__()
        self.linear = nn.Linear(node_embed_dim * veh_fea_dim, output_dim) 

    def forward(self, node_embed, veh_fea):

        batch_size, N, node_embed_dim = node_embed.shape
        num_vehicles, veh_fea_dim = veh_fea.shape[1], veh_fea.shape[2]

        node_embed_exp = node_embed.unsqueeze(2)  
        veh_fea_exp = veh_fea.unsqueeze(1) 

        bilinear_product = torch.einsum('bnid,bvje->bnjde', node_embed_exp, veh_fea_exp) 

        bilinear_flat = bilinear_product.reshape(batch_size * N * num_vehicles, -1) 

        bilinear_output = self.linear(bilinear_flat)

        bilinear_output = bilinear_output.view(batch_size, N, num_vehicles, -1)
        return bilinear_output