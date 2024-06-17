#!/bin/bash

#SBATCH -J ldm_3d_dim64_test_out               # Job name
#SBATCH -o out.ldm_3d_dim64_test_out.%j        # Name of stdout output file (%j expands to %jobId)
#SBATCH -p gpu                           # queue or partiton name
#SBATCH --gres=gpu:1                     # Num Devices

echo
python3 -V
which python3
which pip3
echo
pip3 list | grep pytorch
echo
python3 'test_ddpm.py' params.batch_size=4 +diffusion.sample_mode='xt_x0' +test_name='20240617 ddpm dim64 output' #hydra.run.dir='../outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'

# End of File.

