import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
import math
pi = math.pi
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%m/%d/%Y %H:%M:%S")

import torch
exp = torch.exp
import numpy as np
import cupy as cp
from cupy.fft import fftn, ifftn
import h5py
import hydra
from omegaconf import DictConfig

from train.ddpm_data_loader import read_data, get_batch
from ldm.models.unet import UNet
from ldm.diffusion import Diffusion
from statistic import divergence_free, rms_vel_mag, dissipation, taylor_micro_scale, kolmogorov_length_scale, kolmogorov_time_scale, reynolds_num, e_spectrum



# Parameters
a = 0; b = 2*pi; L = b - a # Box size
N = 128; Nx, Ny, Nz = N, N, N  # Grid resolution
Mx, My, Mz = 3*Nx//2, 3*Ny//2, 3*Nz//2
dt = 0.0025; T = 5*10**1; Nt = int(T/dt); K = 20; del_t = 2*K*dt  # Time step

# Grid and wavenumbers
# x = torch.linspace(a, b, Nx+1, dtype=torch.float32); dx = L/nx
# y = torch.linspace(a, b, Ny+1, dtype=torch.float32); dy = L/ny
# z = torch.linspace(a, b, Nz+1, dtype=torch.float32); dz = L/nz
x, dx = cp.linspace(a, b, Nx, endpoint=False, retstep=True, dtype=np.float32)
y, dy = cp.linspace(a, b, Ny, endpoint=False, retstep=True, dtype=np.float32)
z, dz = cp.linspace(a, b, Nz, endpoint=False, retstep=True, dtype=np.float32)
Z, Y, X = cp.meshgrid(z, y, x, indexing='ij')

kx = cp.fft.fftfreq(Nx, 1/Nx)
ky = cp.fft.fftfreq(Ny, 1/Ny)
kz = cp.fft.fftfreq(Nz, 1/Nz)
KZ, KY, KX = cp.meshgrid(kz, ky, kx, indexing='ij')

k_sq = KX**2 + KY**2 + KZ**2
k_round = cp.round(cp.sqrt(k_sq)).astype(np.int32); k_max = cp.max(k_round)
k_index, k_count = cp.unique(k_round, return_counts=True)
k_sq[0, 0, 0] = 1  # Avoid division by zero

# Physical parameters
nu = 0.006416 # 1.0e-3  # Kinematic viscosity (adjust as needed)



# change config_path depending on the configuration of the trained model
# @hydra.main(version_base=None, config_path="../outputs/2024-06-14/12-48-12/.hydra", config_name="config") #dim32
@hydra.main(version_base=None, config_path="../outputs/2024-06-14/12-48-40/.hydra", config_name="config") #dim64
def test(cfg: DictConfig):
    device = cfg.device; start = cfg.data.start; nt = cfg.data.lead_time
    file_dir = cfg.paths.file_dir; run_name = cfg.run_name; test_name = cfg.test_name

    _, data_val, data_test = read_data(cfg)
    data_test = torch.cat((data_val, data_test), dim=0); SL = data_test.shape[0]
    del data_val
    # model = UNet(cfg).to(device)
    # Restore_Dir = os.path.join(file_dir, "MODELS", run_name, f"{cfg.params.epochs}ckpt.pt")
    # model.load_state_dict(torch.load(Restore_Dir))
    # diffusion = Diffusion(model, cfg); T = diffusion.timesteps
    # logging.info(f"Loading trained model and data.")

    # model.eval()
    # sample_mode = cfg.diffusion.sample_mode
    # with torch.no_grad():
    #     test_dataloader = get_batch(data_test, cfg, shuffle=False)
    #     for it, (X, Y) in enumerate(test_dataloader):
    #         x0 = X.to(device); condition = Y.to(device)
    #         x0 = diffusion.vqgan.encode(x0, quantize=False)
    #         condition = diffusion.vqgan.encode(condition, quantize=False)
    #         # normalize to -1 and 1
    #         x0 = ((x0 - diffusion.vqgan.codebook.embeddings.min()) /
    #                 (diffusion.vqgan.codebook.embeddings.max() -
    #                 diffusion.vqgan.codebook.embeddings.min())) * 2.0 - 1.0
    #         condition = ((condition - diffusion.vqgan.codebook.embeddings.min()) /
    #                 (diffusion.vqgan.codebook.embeddings.max() -
    #                 diffusion.vqgan.codebook.embeddings.min())) * 2.0 - 1.0

    #         x_t = torch.randn_like(x0)
    #         x0_hat = diffusion.sample(x_t, T, condition, sample_mode=sample_mode)
    #         if it == 0:
    #             pred = x0_hat.to(device='cpu')
    #         else:
    #             pred = torch.cat((pred,x0_hat.to(device='cpu')), dim=0)
    # logging.info(f"Inference for test data is done.")
    # del X, Y, x0, condition, x0_hat

    # logging.info(f"Saving prediction results.")
    # Save_path = os.path.join(file_dir, run_name, test_name); os.makedirs(Save_path, exist_ok=True)
    # for q in range(SL-nt):
    #     Save_pred = os.path.join(Save_path, f"Train{cfg.data.num_train} time{(q+start+nt)*del_t:.1f} prediction.h5")
    #     with h5py.File(Save_pred, 'w') as fc:
    #         fc.create_dataset(f"vel{(q+start+nt)*del_t:.1f}", data=pred[q:q+1])
            
    # Filewrite = os.path.join(file_dir, run_name, f"{test_name} Train{cfg.data.num_train} z-pi vis.plt")
    # with open(Filewrite, 'w') as fvis:
    #     fvis.write('VARIABLES="x/pi","y/pi","u","v","w"\n')
    #     for q in range(SL-nt):
    #         fvis.write(f'ZONE T="T={(q+start+nt)*del_t:.1f}" I={Nx} J={Ny}\n')
    #         for j in range(Ny):
    #             for i in range(Nx):
    #                 fvis.write('%lf %lf %lf %lf %lf\n' %(x[i]/pi,y[j]/pi,pred[q,0,Nz//2,j,i],pred[q,1,Nz//2,j,i],pred[q,2,Nz//2,j,i]))
    # logging.info(f"Visualization is done.")

    # when using already sampled prediction results again
    pred = torch.zeros([SL-nt, 3, Nz, Ny, Nx])
    Read_path = os.path.join(file_dir, run_name, test_name)
    for q in range(SL-nt):    
        Read_pred = os.path.join(Read_path, f"Train{cfg.data.num_train} time{(q+start+nt)*del_t:.1f} prediction.h5")
        with h5py.File(Read_pred, 'r') as fr:
            key = f"vel{(q+start+nt)*del_t:.1f}"
            pred[q:q+1] = torch.from_numpy(fr[key][:])

    U_rms = cp.zeros(SL-nt, dtype=np.float32)
    TKE = cp.zeros(SL-nt, dtype=np.float32)
    eddy_turnover = cp.zeros(SL-nt, dtype=np.float32)
    d = cp.zeros(SL-nt, dtype=np.float32)
    lambda_g = cp.zeros(SL-nt, dtype=np.float32)
    eta = cp.zeros(SL-nt, dtype=np.float32)
    tau_eta = cp.zeros(SL-nt, dtype=np.float32)
    Re = cp.zeros(SL-nt, dtype=np.float32)
    E_spectrum = cp.zeros([SL-nt,k_max.get()+1], dtype=np.float32)
    E_spectrum_tar = cp.zeros([SL-nt,k_max.get()+1], dtype=np.float32)
    div_pred = np.zeros([SL-nt, Nz, Ny, Nx], dtype=np.float32)
    div_tar = np.zeros([SL-nt, Nz, Ny, Nx], dtype=np.float32)
    target = data_test[nt:]
    for q in range(SL-nt):
        u, v, w = cp.asarray(pred[q,0]), cp.asarray(pred[q,1]), cp.asarray(pred[q,2])
        uk = fftn(u, axes=(-3,-2,-1), norm='forward')
        vk = fftn(v, axes=(-3,-2,-1), norm='forward')
        wk = fftn(w, axes=(-3,-2,-1), norm='forward')

        U_rms[q] = rms_vel_mag(u, v, w)
        TKE[q], eddy_turnover[q] = 3/2*U_rms[q]**2, L/U_rms[q]
        d[q] = dissipation(uk, vk, wk, KX, KY, KZ, nu)
        lambda_g[q], eta[q], tau_eta[q] = taylor_micro_scale(U_rms[q], d[q], nu), kolmogorov_length_scale(d[q], nu), kolmogorov_time_scale(d[q], nu)
        Re[q] = reynolds_num(U_rms[q], lambda_g[q], nu)
        E_spectrum[q,:] = e_spectrum(uk, vk, wk, k_round, k_sq, k_index)
        div_pred[q] = (divergence_free(uk, vk, wk, KX, KY, KZ)).get()
                
        u, v, w = cp.asarray(target[q,0]), cp.asarray(target[q,1]), cp.asarray(target[q,2])
        uk = fftn(u, axes=(-3,-2,-1), norm='forward')
        vk = fftn(v, axes=(-3,-2,-1), norm='forward')
        wk = fftn(w, axes=(-3,-2,-1), norm='forward')
        E_spectrum_tar[q,:] = e_spectrum(uk, vk, wk, k_round, k_sq, k_index)
        div_tar[q] = (divergence_free(uk, vk, wk, KX, KY, KZ)).get()
    logging.info(f"Calculating statiscs is done.")

    Filewrite = os.path.join(file_dir, run_name, f"{test_name} Train{cfg.data.num_train} statistics.plt")
    with open(Filewrite, 'w') as fstat:
        fstat.write('VARIABLES="t","U_rms","TKE","dissipation","Taylor scale","Re","Kolmo length","Kolmo time","eddy turnover"\n')
        fstat.write(f'Zone T="T=Stats"\n')
        for q in range(SL-nt):
            fstat.write(f'{(q+start+nt)*del_t:.1f} {U_rms[q]} {TKE[q]} {d[q]} {lambda_g[q]} {Re[q]} {eta[q]} {tau_eta[q]} {eddy_turnover[q]}\n')
    
    Filewrite = os.path.join(file_dir, run_name, f"{test_name} Train{cfg.data.num_train} divergenece cond z-pi vis.plt")
    with open(Filewrite, 'w') as fd:
        fd.write('VARIABLES="x/pi","y/pi","Target","Prediction"\n')
        for q in range(SL-nt):
            fd.write(f'Zone T="T={(q+start+nt)*del_t:.1f}" I={Nx} J={Ny}\n')
            for j in range(Ny):
                for i in range(Nx):
                    fd.write('%lf %lf %lf %lf\n' %(x[i]/pi,y[j]/pi,div_tar[q,Nz//2,j,i],div_pred[q,Nz//2,j,i]))

    div_tar = div_tar.mean(axis=(-3,-2,-1))
    div_pred = div_pred.mean(axis=(-3,-2,-1))
    Filewrite = os.path.join(file_dir, run_name, f"{test_name} Train{cfg.data.num_train} divergenece cond spatial mean.plt")
    with open(Filewrite, 'w') as fd:
        fd.write('VARIABLES="t","Target","Prediction"\n')
        fd.write(f'Zone T="T=divergence cond"\n')
        for q in range(SL-nt):
            fd.write(f'{(q+start+nt)*del_t:.1f} {div_tar[q]} {div_pred[q]}\n')

    Filewrite = os.path.join(file_dir, run_name, f"{test_name} Train{cfg.data.num_train} energy spectrum.plt")
    with open(Filewrite, 'w') as fE:
        fE.write('VARIABLES="k","Target","Prediction"\n')
        for q in range(SL-nt):
            fE.write(f'Zone T="T={(q+start+nt)*del_t:.1f}"\n')
            for i in k_index.get():
                fE.write('%d %.12lf %.12lf\n' %(i,E_spectrum_tar[q,i],E_spectrum[q,i]))


    tar_mean = torch.mean(target, dim=(-3,-2,-1), keepdims=True)
    tar_fluc = target - tar_mean
    tar_rms = torch.std(tar_fluc, dim=(-3,-2,-1), correction=0)

    pred_mean = torch.mean(pred, dim=(-3,-2,-1), keepdims=True)
    pred_fluc = pred - pred_mean
    pred_rms = torch.std(pred_fluc, dim=(-3,-2,-1), correction=0)

    corr_coeff = torch.mean(tar_fluc*pred_fluc, dim=(-3,-2,-1))/(tar_rms*pred_rms)
    corr_coeff = torch.mean(corr_coeff, dim=1)
    error = pred - target
    MSE = torch.mean(error**2, dim=(-4,-3,-2,-1))

    Filewrite = os.path.join(file_dir, run_name, f"{test_name} prediction uvw avg CC.plt")
    fw = open(Filewrite, 'w')
    fw.write('VARIABLES="t","Corr_Coef"\n')
    fw.write('ZONE T="correlation coefficient"\n')
    for q in range(SL-nt):
        fw.write('%.2lf %.10lf\n' %((q+start+nt)*del_t,corr_coeff[q]))
    fw.close()

    Filewrite = os.path.join(file_dir, run_name, f"{test_name} prediction uvw avg MSE.plt")
    fw = open(Filewrite, 'w')
    fw.write('VARIABLES="t","MSE"\n')
    fw.write('ZONE T="Mean Square Error"\n')
    for q in range(SL-nt):
        fw.write('%.2lf %.10lf\n' %((q+start+nt)*del_t,MSE[q]))
    fw.close()

if __name__ == '__main__':
    test()