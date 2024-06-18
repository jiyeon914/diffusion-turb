#!/bin/bash

#SBATCH -J ldm_3d_nt6_dim64_hubersum               # Job name
#SBATCH -o out.ldm_3d_nt6_dim64_hubersum.%j        # Name of stdout output file (%j expands to %jobId)
#SBATCH -p gpu                           # queue or partiton name
#SBATCH --gres=gpu:1                     # Num Devices

echo
python3 -V
which python3
which pip3
echo
pip3 list | grep pytorch
echo
python3 'train_ddpm.py' run_name='20240618 ldm nt6 dim64 huber sum' params.batch_size=8 params.epochs=500 unet.dim=64 unet.init_dim=32 diffusion.loss_reduction='none' #  data.lead_time=3

# End of File.

