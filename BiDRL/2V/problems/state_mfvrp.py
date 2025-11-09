import torch
from typing import NamedTuple
import numpy as np


cap = [4.6,3.2]
cap = torch.tensor(cap)
energy=[120.,106.]
energy=torch.tensor(energy)
start_cost= [80,73]
start_cost=torch.tensor(start_cost)

class StateMFVRP:

    def __init__(self, coords, demand, ids, veh, pre_a, used_capacity, surplus_capacity, lengths, surplus_energy,
                 cur_time, cur_Ecost, visited_, cur_coord, cur_veh, i, early_time, last_time, serve_time, mfvrp_size, num_station):
        self.coords = coords
        self.demand = demand 
        self.ids = ids
        self.veh = veh
        self.pre_a = pre_a 
        self.used_capacity = used_capacity
        self.surplus_capacity = surplus_capacity
        self.lengths = lengths
        self.surplus_energy = surplus_energy
        self.cur_time = cur_time
        self.cur_Ecost = cur_Ecost
        self.visited_ = visited_
        self.cur_coord = cur_coord
        self.cur_veh = cur_veh
        self.i = i 
        self.early_time = early_time
        self.last_time = last_time
        self.serve_time = serve_time
        self.mfvrp_size = mfvrp_size
        self.num_station = num_station


    @staticmethod
    def initialize(input):
        coords = input[:, :, :2] 
        demand = input[:, :, 2]
        timewindow=input[:, :, 3:5] 
        serve_time = input[:, :, 5]  
        batch_size ,N= serve_time.size()
        depot =coords[:, 0, :] 
        early_time =timewindow[:,:,0] 
        last_time = timewindow[:,:,1]
        mfvrp_size = round((N-1)*1/1.1)
        num_station = int(mfvrp_size/10)

        return StateMFVRP(
            coords = coords,
            demand = demand, 
            ids =torch.arange(batch_size,dtype=torch.int,device=coords.device).unsqueeze(1), 
            veh=torch.arange(len(cap),dtype=torch.int,device=coords.device).unsqueeze(1),
            pre_a = torch.zeros((batch_size,1),dtype=torch.int,device=coords.device),
            used_capacity=demand.new_zeros((batch_size,1),dtype=torch.int,device=coords.device),
            surplus_capacity = demand.new_zeros((batch_size,1),dtype=torch.int,device=coords.device),
            lengths = torch.zeros(batch_size,1,device=coords.device),
            surplus_energy = energy.new_zeros(batch_size,1,device=coords.device), 
            cur_time = torch.zeros((batch_size,1),dtype=torch.int,device=coords.device),
            cur_Ecost = torch.zeros((batch_size,1),dtype=torch.float,device=coords.device),
            visited_=torch.zeros((batch_size, N),dtype=torch.uint8,device=coords.device), 
            cur_coord=depot,
            cur_veh = torch.zeros((batch_size,1),dtype=torch.int,device=coords.device),
            i=torch.zeros(1, dtype=torch.int,device=coords.device),
            early_time=  early_time,
            last_time= last_time,
            serve_time= serve_time,
            mfvrp_size = mfvrp_size,
            num_station = num_station
        )


    def update(self, selected, veh): 
        device = self.coords.device
        selected = torch.tensor(selected) 
        consume = torch.tensor([0.0426,0.1436]).to(device) 
        weight = torch.tensor([3.4,4.1]).to(device)
        cap = [4.6,3.2]
        cap = torch.tensor(cap).to(device)
        energy=[120.,106.]
        energy=torch.tensor(energy).to(device)
        wage = 10
        speed_veh = torch.tensor([1.1,1]).to(device)
        charge_rate = 1.6 
        prize = [7.2,1.3]
        prize = torch.tensor(prize).to(device)

        all_zeros = torch.all(selected == 0)  
        if all_zeros:
            batch_size, _ = selected.size()  
            pre_a = selected
            used_capacity=self.demand.new_zeros((batch_size,1),dtype=torch.int)
            all_weight = self.used_capacity+weight[veh]
            surplus_capacity=self.demand.new_zeros((batch_size,1),dtype=torch.int) 
            cur_coord=self.coords[:,0,:].to(device)
            distance_tow_node = (cur_coord - self.cur_coord).norm(p=2, dim=-1).unsqueeze(1)
            lengths = self.lengths+distance_tow_node
            surplus_energy = energy.new_zeros(batch_size,1)  
            cur_time = (((self.cur_time.clone()).view(512,1)).float()).to(device)
            last_time = cur_time
            cur_time = (self.cur_time + distance_tow_node/speed_veh[veh]).to(device)
            work_time = ((torch.ceil(cur_time/60))-(torch.ceil(last_time/60))).to(device)
            cur_Ecost = self.cur_Ecost+all_weight * distance_tow_node * consume[veh]*prize[veh]+wage*work_time
            visited_ = self.visited_
            
            cur_veh = torch.zeros((batch_size,1),dtype=torch.int) 
            i=torch.zeros(1, dtype=torch.int),

            self.pre_a = pre_a 
            self.used_capacity = used_capacity 
            self.surplus_capacity = surplus_capacity
            self.lengths = lengths 
            self.surplus_energy = surplus_energy 
            self.cur_time = cur_time 
            self.cur_Ecost = cur_Ecost
            self.visited_ = visited_ 
            self.cur_coord = cur_coord 
            self.cur_veh = cur_veh 
            self.i = 0
                                
            return self

        else:
            pre_a = selected   
            batch_size, _ = selected.size()  
            demand= self.demand.clone()
            demand[torch.arange(batch_size),selected.squeeze(1)]=0 
            all_weight = self.used_capacity+weight[veh]
            used_capacity =(self.used_capacity+(self.demand[torch.arange(batch_size),selected.squeeze(1)]).unsqueeze(1)).float()  

            surplus_capacity = self.surplus_capacity-(self.demand[torch.arange(batch_size),selected.squeeze(1)]).unsqueeze(1)
            cur_coord = self.coords[torch.arange(batch_size),selected.squeeze(1),:] 

            distance_tow_node = ((cur_coord - self.cur_coord).norm(p=2, dim=-1).unsqueeze(1)).float()
            lengths = self.lengths + distance_tow_node
            conditionA =  ((veh == 0)) & (selected <= self.mfvrp_size)
            conditionB =  (veh == 1) 
            conditionC =(selected >= 1) & (selected <= self.mfvrp_size) 
            conditionI = selected==0 

            maskA = conditionA
            maskB = conditionB & conditionC 
            maskBC = conditionB & ~conditionC  

            surplus_energy = self.surplus_energy.clone()
            surplus_energy = surplus_energy.float()
            surplus_energy[maskA] = surplus_energy[maskA]-distance_tow_node[maskA]*all_weight[maskA]*consume[veh][maskA] 
            surplus_energy[maskBC] = energy[veh][maskBC]
            surplus_energy[maskB] = surplus_energy[maskB]-distance_tow_node[maskB]*all_weight[maskB]*consume[veh][maskB]  

            cur_time = (((self.cur_time.clone()).view(512,1)).float()).to(device)
            last_time = self.cur_time.view(512,1).float().to(device)

            conditionD = conditionA | (conditionB & conditionC)

            maskBCI = conditionB & ~conditionC & conditionI 
            maskBC_I = conditionB & ~conditionC & ~conditionI  

            maskD = conditionD
            early_time_selected = ((self.early_time[torch.arange(batch_size),selected.squeeze(1)]).unsqueeze(1)).float()
            serve_time_selected = ((self.serve_time[torch.arange(batch_size),selected.squeeze(1)]).unsqueeze(1)).float()

            conditionF = torch.zeros_like(maskD, dtype=torch.bool)
            conditionF[maskD] = (cur_time[maskD]+distance_tow_node[maskD]/speed_veh[veh[maskD]] <= early_time_selected[maskD])

            conditionG = conditionD & conditionF   
            conditionH = conditionD & ~conditionF 

            maskG= conditionG
            maskH= conditionH

            cur_time[maskG] = early_time_selected[maskG]+serve_time_selected[maskG]
            cur_time[maskH] =cur_time[maskH] + distance_tow_node[maskH]/speed_veh[veh[maskH]] +serve_time_selected[maskH]

            cur_time[maskBC_I] = cur_time[maskBC_I] + distance_tow_node[maskBC_I]/speed_veh[veh[maskBC_I]]+(
                        energy[veh][maskBC_I]-
                        (self.surplus_energy[maskBC_I]-(all_weight[maskBC_I]*distance_tow_node[maskBC_I]*consume[veh][maskBC_I])))/charge_rate
            cur_time[maskBCI] = cur_time[maskBCI] + distance_tow_node[maskBCI]/speed_veh[veh[maskBCI]]
    
            work_time = ((torch.ceil(cur_time/60))-(torch.ceil(last_time/60))).to(device)
            
            cur_Ecost = self.cur_Ecost +all_weight * distance_tow_node * consume[veh]*prize[veh]+wage*work_time  
            go_station = selected > self.mfvrp_size
            cur_Ecost = cur_Ecost +go_station.float() * 20
            visited_ = self.visited_.clone()
            visited_[torch.arange(batch_size),selected.squeeze(1)] =1 
            num_station = int(self.num_station)
            visited_[:, 0] = 0
            
            cur_veh = veh
            
            self.demand = demand
            self.pre_a = selected  
            self.used_capacity = used_capacity 
            self.surplus_capacity = surplus_capacity 
            self.lengths = lengths 
            self.surplus_energy = surplus_energy 
            self.cur_time = cur_time 
            self.cur_Ecost = cur_Ecost
            self.visited_ = visited_ 
            self.cur_coord = cur_coord 
            self.cur_veh = cur_veh 
            self.i = self.i +1
                                
            return self
        

    def not_all_finished(self):
        m = self.mfvrp_size
        visited = self.visited_
        sub_tensor = visited[:, 1:m+1]
        cur_node = self.pre_a
        condotion2 = torch.all(cur_node == 0)
        condition1 = torch.all(sub_tensor == 1)
        condition3 = ~(condition1 & condotion2)
        return condition3 

    def get_finished(self): 
        return self.visited_.sum(-1) == self.visited_.size(-1)-1-self.num_station

    def get_current_node(self):
        return self.pre_a

    def construct_solutions(self,actions):
        return actions




    
            
           

        
