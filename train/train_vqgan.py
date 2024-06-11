import sys
import os
import math
pi = math.pi
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%m/%d/%Y %H:%M:%S")

import torch
from torch import nn
from torch import optim
exp = torch.exp
from torchinfo import summary

from vq_data_loader import *
from vq_gan_3d.models.vq_gan import VQGAN



# Parameters
a = 0; b = 2*pi; L = b - a # Box size
N = 128; Nx, Ny, Nz = N, N, N  # Grid resolution
Mx, My, Mz = 3*Nx//2, 3*Ny//2, 3*Nz//2
dt = 0.0025; T = 5*10**1; Nt = int(T/dt); K = 20; del_t = 2*K*dt  # Time step

# Grid and wavenumbers
x, dx = np.linspace(a, b, Nx, retstep=True, dtype=np.float32)
y, dy = np.linspace(a, b, Ny, retstep=True, dtype=np.float32)
z, dz = np.linspace(a, b, Nz, retstep=True, dtype=np.float32)
Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

kx = np.fft.fftfreq(Nx, 1/Nx)
ky = np.fft.fftfreq(Ny, 1/Ny)
kz = np.fft.fftfreq(Nz, 1/Nz)
KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')

k_sq = KX**2 + KY**2 + KZ**2
k_round = np.round(np.sqrt(k_sq)).astype(np.int32); k_max = np.max(k_round)
k_index, k_count = np.unique(k_round, return_counts=True)
k_sq[0, 0, 0] = 1  # Avoid division by zero

# Physical parameters
nu = 0.006416 # 1.0e-3  # Kinematic viscosity (adjust as needed)



def setup_logging(Filename, run_name):
    os.makedirs(os.path.join(Filename, "MODELS", run_name), exist_ok=True)
    os.makedirs(os.path.join(Filename, run_name), exist_ok=True)

def train(args):
    Filename = args.Filename; run_name = args.run_name
    setup_logging(Filename, run_name)

    device = args.device; data_res = args.data_res; batch_size = args.batch_size
    data_train, data_val, _ = read_data(args)

    model = VQGAN(dim=64, init_dim=32, dim_mults=(1, 2, 4), z_channels=3).to(device)
    summary(model, input_size = [(1, 3, 128, 128, 128)])
    print("VQ-GAN : ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(model)
    print(f'Allocated: {torch.cuda.memory_allocated()/1024**3} GB')
    print(f'Reserved: {torch.cuda.memory_reserved()/1024**3} GB')
    torch.cuda.empty_cache()

    opt_ae = optim.Adam(list(model.encoder.parameters()) +
                        list(model.decoder.parameters()) +
                        list(model.pre_vq_conv.parameters()) +
                        list(model.post_vq_conv.parameters()) +
                        list(model.codebook.parameters()),
                        lr=args.lr, betas=(0.5, 0.9))
    opt_disc = optim.Adam(#list(model.image_discriminator.parameters()) +
                          list(model.video_discriminator.parameters()),
                          lr=args.lr, betas=(0.5, 0.9))
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.5)

    Filewrite1 = os.path.join(Filename, run_name, f"{run_name} training loss.plt")
    fMSE = open(Filewrite1, 'w')
    fMSE.write('VARIABLES="epoch","AE","D"\n')
    fMSE.write('Zone T="training loss"\n')
    fMSE.close()

    torch.cuda.empty_cache()
    for epoch in range(1,args.epochs+1):
        logging.info(f"Starting epoch {epoch}:")
        model.train()
        train_dataloader = get_batch(data_train, args, shuffle=True); l = len(train_dataloader); print(l)
        for iter, X in enumerate(train_dataloader):
            X = X.to(device)

            # Train autoencoder
            opt_ae.zero_grad()
            recon_loss, vq_output, aeloss, perceptual_loss, gan_feat_loss = model.compute_loss(X, optimizer_idx=0)
            # recon_loss, vq_output, aeloss, gan_feat_loss = model.compute_loss(X, optimizer_idx=0)
            loss_ae = recon_loss + vq_output['commitment_loss'] + aeloss + perceptual_loss + gan_feat_loss
            # loss_ae = recon_loss + vq_output['commitment_loss'] + aeloss + gan_feat_loss
            loss_ae.backward()
            opt_ae.step()

            # Train discriminator
            opt_disc.zero_grad()
            loss_disc = model.compute_loss(X, optimizer_idx=1)
            loss_disc.backward()
            opt_disc.step()

            model.update_step()
            if iter % (l//5) == 0: logging.info(f"Iter {iter}, loss ae {loss_ae.item()}, loss D {loss_disc.item()}")

            if epoch == 1 and iter == 0:
                fMSE = open(Filewrite1, 'a')
                fMSE.write('%d %.10f %.10f\n' %(epoch-1, loss_ae.item(), loss_disc.item()))
                fMSE.close()
        # scheduler.step()
        logging.info(f"End of epoch{epoch} global step {model.global_step}: loss ae {loss_ae.item():.10f}, loss D {loss_disc.item()}")

        fMSE = open(Filewrite1, 'a')
        fMSE.write('%d %.10f %.10f\n' %(epoch, loss_ae.item(), loss_disc.item()))
        fMSE.close()

        if epoch % (args.epochs//5) == 0:
            torch.save(model.state_dict(), os.path.join(Filename, "MODELS", run_name, f"{epoch}ckpt.pt"))

            model.eval()
            with torch.no_grad():
                val_dataloader = get_batch(data_val, args, shuffle=False)
                for it, X in enumerate(val_dataloader):
                    X = X.to(device)
                    # recon_val_loss, _, _, _ = model.compute_loss(X, optimizer_idx=0)
                    # val_loss += recon_val_loss.item()
                    
                    x_recon, _ = model(X)
                    if it == 0:
                        pred = x_recon.to(device='cpu')
                    else:
                        pred = torch.cat((pred,x_recon.to(device='cpu')), dim=0)

            Filewrite2 = os.path.join(Filename, run_name, f"{run_name} val z-pi vis epoch={epoch}.plt")
            fvis = open(Filewrite2, 'w')
            fvis.write('VARIABLES="x/pi","y/pi","u","v","w"\n')
            for q in range(data_val.shape[0]):
                fvis.write(f'ZONE T="T={(q+args.start)*del_t:.1f}" I={Nx} J={Ny}\n')
                for j in range(Ny):
                    for i in range(Nx):
                        fvis.write('%lf %lf %lf %lf %lf\n' %(x[i]/pi,y[j]/pi,pred[q,0,Nz//2,j,i],pred[q,1,Nz//2,j,i],pred[q,2,Nz//2,j,i]))
            fvis.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args = [])
    args.run_name = "20240528 LDM 3D vqgan test"
    args.Filename = '/data/jykim3994/4.LDM'
    args.Data_Dir = '/data/jykim3994/0.Data/3D FHIT/resolution 128'
    args.data_res = 128
    args.seq_length = Nt//(2*K) + 1
    args.start = 100
    args.epochs = 200
    args.batch_size = 1
    args.lr = 1e-4
    args.device = "cuda"

    train(args)
