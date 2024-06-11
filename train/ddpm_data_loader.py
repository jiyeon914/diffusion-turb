import numpy as np
import h5py
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from einops import rearrange, reduce
import math
pi = math.pi; exp = torch.exp

def read_data(args):
    nt = args.lead_time; Data_Dir = args.Data_Dir; N = args.data_res; Nx, Ny, Nz = N, N, N
    start = args.start; seq_length = args.seq_length
    train_num = args.train_num; test_num = args.test_num
    dt = 0.0025; T = 5*10**1; Nt = int(T/dt); K = 20; del_t = 2*K*dt; nu = 0.006416

    data_vel = np.zeros([seq_length,3,Nz,Ny,Nx], dtype = np.float64)
    for q in range(start,seq_length):
        ### Read data ###
        ReadData = Data_Dir + f"/Training data/Train{0} time{q*del_t:.1f} 3D FHIT nu={nu} n={N} T={T} dt={dt:.4f} K={2*K} data.h5"
        fr = h5py.File(ReadData, 'r')
        key = f"vel{q*del_t:.1f}" if q > 0 else f"vel{0}"
        data_vel[q:q+1,:,:,:,:] = fr[key][:]
        fr.close()
    data_vel = torch.from_numpy(data_vel[start:,:,:,:,:])

    data_train = data_vel[:train_num+nt,:,:,:,:]; data_test = data_vel[-test_num-nt:,:,:,:,:]
    return data_train, data_test

def get_batch(data, args, shuffle=True):
    nt = args.lead_time; batch_size = args.batch_size; length = data.shape[0]

    data = data.to(torch.float32)
    data /= torch.std(data, dim=(-2,-1), correction=0, keepdim=True)
    input_field = rearrange(data[:length-nt,:,:,:,:], "t c z y x -> (t z) c y x")
    target = rearrange(data[nt:,:,:,:,:], "t c z y x -> (t z) c y x")
    data_cat = TensorDataset(input_field,target)
    dataloader = DataLoader(data_cat, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader

def read_data_res128(args):
    num_train = 500; num_val = 100; num_test = 50
    Data_Dir = args.Data_Dir; nx = ny = args.data_res

    ### Read training data ###
    ReadTrain = Data_Dir + '/2D_DHIT_TRAINING.h5'
    fr = h5py.File(ReadTrain, 'r')
    data_train = np.zeros([num_train,ny,nx,1], dtype = np.float64)
    for t in range(num_train):
        vol = f"Simul{t} Vol_000"
        data_train[t,:] = fr[vol][:]
    fr.close()
    data_train = torch.from_numpy(data_train)
    data_train = data_train.transpose(1,2).transpose(1,3)

    # ### Read validation data ###
    ReadVal = Data_Dir + '/2D_DHIT_VALIDATION.h5'
    fr = h5py.File(ReadVal, 'r')
    data_val = np.zeros([num_val,ny,nx,1], dtype = np.float64)
    for t in range(num_val):
        vol = f"Simul{t} Vol_000"
        data_val[t,:] = fr[vol][:]
    fr.close()
    data_val = torch.from_numpy(data_val)
    data_val = data_val.transpose(1,2).transpose(1,3)

    # ### Read test data ###
    ReadTest = Data_Dir + '/2D_DHIT_TEST.h5'
    fr = h5py.File(ReadTest, 'r')
    data_test = np.zeros([num_test,ny,nx,1], dtype = np.float64)
    for t in range(num_test):
        vol = f"Simul{t} Vol_000"
        data_test[t,:] = fr[vol][:]
    fr.close()
    data_test = torch.from_numpy(data_test)
    data_test = data_test.transpose(1,2).transpose(1,3)

    scaling_factor = torch.std(data_train, dim = None, correction = 0)
    data_train /= scaling_factor
    data_val /= scaling_factor
    data_test /= scaling_factor
    return data_train, data_val, data_test, scaling_factor

def get_vel(vol, args):
    data_res = args.data_res; nx = ny = data_res; a = 0; b = 2*pi; L = b - a

    kx = [(2*pi/L)*px for px in range(nx//2+1)]
    kx = torch.unsqueeze(torch.DoubleTensor(kx), dim = 0)
    ky = [(2*pi/L)*py if py < ny/2 else (2*pi/L)*py-ny for py in range(ny)]
    ky = torch.unsqueeze(torch.DoubleTensor(ky), dim = -1)
    k_mag = kx**2 + ky**2
    k_cal = torch.reshape(k_mag, [1,1,ny,nx//2+1])

    psik = torch.zeros([vol.shape[0],1,ny,nx//2+1], dtype = torch.complex128)
    volk = torch.fft.rfft2(vol, dim = (2,3))/(nx*ny)

    psik[:,:,0:,1:] = volk[:,:,0:,1:]/k_cal[:,:,0:,1:]
    psik[:,:,1:,0:1] = volk[:,:,1:,0:1]/k_cal[:,:,1:,0:1]

    psi_xk = 1J*kx*psik; psi_yk = 1J*ky*psik
    psi_x = torch.fft.irfft2(psi_xk, dim = (2,3))*(nx*ny); psi_y = torch.fft.irfft2(psi_yk, dim = (2,3))*(nx*ny)
    return psi_y, -psi_x
