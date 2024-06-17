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
import hydra
from omegaconf import DictConfig

from train.vq_data_loader import read_data, get_batch
from vq_gan_3d.models.vq_gan import VQGAN
from statistic import rms_vel_mag, dissipation, taylor_micro_scale, kolmogorov_length_scale, kolmogorov_time_scale, reynolds_num, e_spectrum



# Parameters
a = 0; b = 2*pi; L = b - a # Box size
N = 128; Nx, Ny, Nz = N, N, N  # Grid resolution
Mx, My, Mz = 3*Nx//2, 3*Ny//2, 3*Nz//2
dt = 0.0025; T = 5*10**1; Nt = int(T/dt); K = 20; del_t = 2*K*dt  # Time step

# Grid and wavenumbers
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
@hydra.main(version_base=None, config_path="../outputs/2024-06-12/11-07-44/.hydra", config_name="config")
def test(cfg: DictConfig):
    device = cfg.device; start = cfg.data.start
    file_dir = cfg.paths.file_dir; run_name = cfg.run_name; test_name = cfg.test_name

    _, data_val, data_test = read_data(cfg); SL = data_val.shape[0] + data_test.shape[0]
    model = VQGAN(cfg).to(device)
    Restore_Dir = os.path.join(file_dir, "MODELS", run_name, f"{cfg.params.epochs}ckpt.pt")
    model.load_state_dict(torch.load(Restore_Dir))
    logging.info(f"Loading trained model and data.")

    model.eval()
    with torch.no_grad():
        val_dataloader = get_batch(data_val, cfg, shuffle=False)
        for it, X in enumerate(val_dataloader):
            X = X.to(device)
            x_recon, _ = model(X)
            if it == 0:
                pred = x_recon.to(device='cpu')
            else:
                pred = torch.cat((pred,x_recon.to(device='cpu')), dim=0)
        
        test_dataloader = get_batch(data_test, cfg, shuffle=False)
        for it, X in enumerate(test_dataloader):
            X = X.to(device)
            x_recon, _ = model(X)
            pred = torch.cat((pred,x_recon.to(device='cpu')), dim=0)
    logging.info(f"Inference for test data is done.")

    Filewrite1 = os.path.join(file_dir, run_name, f"{test_name} Train{cfg.data.num_train} z-pi vis.plt")
    fvis = open(Filewrite1, 'w')
    fvis.write('VARIABLES="x/pi","y/pi","u","v","w"\n')
    for q in range(SL):
        fvis.write(f'ZONE T="T={(q+start)*del_t:.1f}" I={Nx} J={Ny}\n')
        for j in range(Ny):
            for i in range(Nx):
                fvis.write('%lf %lf %lf %lf %lf\n' %(x[i]/pi,y[j]/pi,pred[q,0,Nz//2,j,i],pred[q,1,Nz//2,j,i],pred[q,2,Nz//2,j,i]))
    fvis.close()
    logging.info(f"Visualization is done.")


    U_rms = cp.zeros(SL, dtype=np.float64)
    TKE = cp.zeros(SL, dtype=np.float64)
    eddy_turnover = cp.zeros(SL, dtype=np.float64)
    d = cp.zeros(SL, dtype=np.float64)
    lambda_g = cp.zeros(SL, dtype=np.float64)
    eta = cp.zeros(SL, dtype=np.float64)
    tau_eta = cp.zeros(SL, dtype=np.float64)
    Re = cp.zeros(SL, dtype=np.float64)
    E_spectrum = cp.zeros([SL,k_max.get()+1], dtype=np.float64)
    E_spectrum_tar = cp.zeros([SL,k_max.get()+1], dtype=np.float64)
    target = torch.cat((data_val, data_test), dim=0)
    for q in range(SL):
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
        
        u, v, w = cp.asarray(target[q,0]), cp.asarray(target[q,1]), cp.asarray(target[q,2])
        uk = fftn(u, axes=(-3,-2,-1), norm='forward')
        vk = fftn(v, axes=(-3,-2,-1), norm='forward')
        wk = fftn(w, axes=(-3,-2,-1), norm='forward')
        E_spectrum_tar[q,:] = e_spectrum(uk, vk, wk, k_round, k_sq, k_index)
    logging.info(f"Calculating statiscs is done.")

    Filewrite1 = os.path.join(file_dir, run_name, f"{test_name} Train{cfg.data.num_train} statistics.plt")
    fstat = open(Filewrite1, 'w')
    fstat.write('VARIABLES="t","U_rms","TKE","dissipation","Taylor scale","Re","Kolmo length","Kolmo time","eddy turnover"\n')
    fstat.write(f'Zone T="T=Avg"\n')
    for q in range(SL):
        fstat.write(f'{(q+start)*del_t:.1f} {U_rms[q]} {TKE[q]} {d[q]} {lambda_g[q]} {Re[q]} {eta[q]} {tau_eta[q]} {eddy_turnover[q]}\n')
    fstat.close()

    Filewrite2 = os.path.join(file_dir, run_name, f"{test_name} Train{cfg.data.num_train} energy spectrum.plt")
    fE = open(Filewrite2, 'w')
    fE.write('VARIABLES="k","Target","Recon"\n')
    for q in range(SL):
        fE.write(f'Zone T="T={(q+start)*del_t:.1f}"\n')
        for i in k_index.get():
            fE.write('%d %.12lf %.12lf\n' %(i,E_spectrum_tar[q,i],E_spectrum[q,i]))
    fE.close()


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

    Filewrite = os.path.join(file_dir, run_name, f"{test_name} reconstruction uvw avg CC.plt")
    fw = open(Filewrite, 'w')
    fw.write('VARIABLES="t","Corr_Coef"\n')
    fw.write('ZONE T="correlation coefficient"\n')
    for q in range(SL):
        fw.write('%.2lf %.10lf\n' %((q+start)*del_t,corr_coeff[q]))
    fw.close()

    Filewrite = os.path.join(file_dir, run_name, f"{test_name} reconstruction uvw avg MSE.plt")
    fw = open(Filewrite, 'w')
    fw.write('VARIABLES="t","MSE"\n')
    fw.write('ZONE T="Mean Square Error"\n')
    for q in range(SL):
        fw.write('%.2lf %.10lf\n' %((q+start)*del_t,MSE[q]))
    fw.close()

if __name__ == '__main__':
    test()