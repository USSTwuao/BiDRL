import numpy as np
import argparse
import os
from dispose.data_dispose import check_extension, save_dataset
import torch


def generate_MFVRP_data(batch_size, mfvrp_size):
    mean=0.5
    std_dev=0.2
    veh_speed = 1  #km/min = 60km/h
    num_station = int(mfvrp_size / 10)
    seed = 6
    rnd = np.random.RandomState(seed)

    #coordinate
    cus_loc=rnd.uniform(0, 100, size=(batch_size, mfvrp_size, 2)).astype(int)   
    depot_loc =rnd.uniform(40,60,size=(batch_size, 2)).astype(int)
    depot_loc=depot_loc[:,None,:]
    station_loc=rnd.uniform(10, 90, size=(batch_size, num_station, 2)).astype(int)
    all_loc=np.concatenate((depot_loc, cus_loc,station_loc), axis=1)  #(batch_size,N,2)

    #demand
    choices = np.round(np.arange(0.1, 1.0, 0.1), decimals=1)
    cus_demand = np.random.choice(choices, [batch_size, mfvrp_size])
    depot_demand=np.zeros((batch_size, 1), dtype=int)
    station_demand=np.zeros((batch_size, num_station), dtype=int)
    all_demand=np.concatenate((depot_demand,cus_demand,station_demand), axis=1)  #(batch_size,1+mfvrp_size+num_station)

    #time window and make sure viable solution
    serve_time=rnd.randint(10,30,[batch_size,mfvrp_size]).astype(int)  #[batch_size , mfvrp_size]
    depot_loc = np.repeat(depot_loc, mfvrp_size, axis=1)  #[batch_size,mfvrp_size,2]
    cus_straight_times_to_depot=np.sqrt(np.sum((cus_loc - depot_loc) ** 2, axis=2)) / veh_speed
    start_times=rnd.randint(0, 1200, (batch_size, mfvrp_size)).astype(int) 


    spans=np.clip(rnd.normal(mean, std_dev, (batch_size, mfvrp_size)), 0, 1)*240+60
    end_times = start_times + spans
    end_times = np.where(end_times > 1410, 1410, end_times).astype(int)  
    end_times = np.where(end_times < cus_straight_times_to_depot, cus_straight_times_to_depot+1, end_times).astype(int) 
        
    depot_time_window=np.zeros((batch_size, 1,2))
    depot_time_window[:,:,1]=2880
    station_time_window=np.zeros((batch_size,num_station,2))
    station_time_window[:,:,1]=2880
    cus_time_window=np.stack((start_times, end_times), axis=2)
    all_time_window=np.concatenate((depot_time_window,cus_time_window,station_time_window), axis=1)
    deopt_ser_time = np.zeros((batch_size, 1))
    station_ser_time = np.zeros((batch_size, num_station))
    all_serve_time = np.concatenate((deopt_ser_time ,serve_time, station_ser_time ),axis =1 )

    all_demand = all_demand[:, :, np.newaxis]
    all_serve_time = all_serve_time[:, :, np.newaxis]

    data = np.concatenate((all_loc, all_demand, all_time_window, all_serve_time), axis=2)
    data = torch.tensor(data)

    return data

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--batch_size", type=int, default=512, help="1/10 Size of the dataset")  
    parser.add_argument("--veh_num", type=int, default=4, help="number of the vehicles; 4")  
    parser.add_argument('--mfvrp_size', type=int, default=50, help="Sizes of problem instances: {30, 50, 70, 90, 110} for 4 vehicles")
    parser.add_argument('--num_station',type=int, default=5, help="Number of station in instances: {3,5,7,9,11}")
    parser.add_argument('--veh_speed',type=float, default=1, help="veh speed")
    opts = parser.parse_args()  

    data_dir='data'
    problem='mfvrp'
    datadir = os.path.join(data_dir, problem) 
    os.makedirs(datadir, exist_ok=True) 
    seed = 5
    np.random.seed(seed)
    filename = os.path.join(datadir, 'test_file{}_v{}_{}_seed{}.pkl'.format(problem, opts.mfvrp_size, opts.num_station, seed))

    dataset = generate_MFVRP_data(1,opts.mfvrp_size)
    save_dataset(dataset,filename)