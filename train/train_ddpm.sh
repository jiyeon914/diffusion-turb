#!/bin/bash

#SBATCH -J ldm_3d_test               # Job name
#SBATCH -o out.ldm_3d_test.%j        # Name of stdout output file (%j expands to %jobId)
#SBATCH -p gpu                           # queue or partiton name
#SBATCH --gres=gpu:1                     # Num Devices

echo
python3 -V
which python3
which pip3
echo
pip3 list | grep pytorch
echo
python3 'train_ddpm.py' run_name='20240614 ldm test dim64' unet.dim=64 unet.init_dim=32 params.batch_size=8

# End of File.

