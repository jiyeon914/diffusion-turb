import numpy as np
import h5py
import torch
exp = torch.exp
from torch.utils.data import TensorDataset, DataLoader
import math
pi = math.pi
from einops import rearrange

def read_data(cfg):
    data_dir = cfg.paths.data_dir; num_train = cfg.data.num_train; val_sample = cfg.data.val_sample
    N = cfg.data.data_res; Nx, Ny, Nz = N, N, N; start = cfg.data.start; 
    dt = 0.0025; T = 5*10**1; Nt = int(T/dt); K = 20; seq_length = Nt//(2*K) + 1; del_t = 2*K*dt; nu = 0.006416

    data_train = np.zeros([num_train,seq_length,3,Nz,Ny,Nx], dtype=np.float32)
    for num in range(num_train):
        for q in range(start,seq_length):
            ### Read data ###
            ReadData = data_dir + f"/Training data/Train{num} time{q*del_t:.1f} 3D FHIT nu={nu} n={N} T={T} dt={dt:.4f} K={2*K} data.h5"
            fr = h5py.File(ReadData, 'r')
            key = f"vel{q*del_t:.1f}" if q > 0 else f"vel{0}"
            data_train[num,q:q+1,:,:,:,:] = fr[key][:]
            fr.close()
    data_train = rearrange(torch.from_numpy(data_train[:,start:]), "s t c d h w -> (s t) c d h w")
    
    data_test = np.zeros([seq_length,3,Nz,Ny,Nx], dtype = np.float32)
    for q in range(start,seq_length):
        ### Read data ###
        ReadData = data_dir + f"/Training data/Train{num+1} time{q*del_t:.1f} 3D FHIT nu={nu} n={N} T={T} dt={dt:.4f} K={2*K} data.h5"
        fr = h5py.File(ReadData, 'r')
        key = f"vel{q*del_t:.1f}" if q > 0 else f"vel{0}"
        data_test[q:q+1,:,:,:,:] = fr[key][:]
        fr.close()
    data_val = torch.from_numpy(data_test[start:start+val_sample])
    data_test = torch.from_numpy(data_test[start+val_sample:])
    print(data_train.shape, data_val.shape, data_test.shape)
    return data_train, data_val, data_test

def get_batch(data, cfg, shuffle=True):
    nt = cfg.data.lead_time; batch_size = cfg.params.batch_size; length = data.shape[0]

    input_field = data[:length-nt] # rearrange(data[:length-nt,:,:,:,:], "t c z y x -> (t z) c y x")
    target = data[nt:] # rearrange(data[nt:,:,:,:,:], "t c z y x -> (t z) c y x")
    data_cat = TensorDataset(target, input_field)
    dataloader = DataLoader(data_cat, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    return dataloader