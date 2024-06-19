#!/bin/bash

#SBATCH -J vq_gan_3d_high_z               # Job name
#SBATCH -o out.vq_gan_3d_high_z.%j        # Name of stdout output file (%j expands to %jobId)
#SBATCH -p gpu                           # queue or partiton name
#SBATCH --gres=gpu:1                     # Num Devices

echo
python3 -V
which python3
which pip3
echo
pip3 list | grep pytorch
echo
python3 'train_vqgan.py' run_name='20240618 vqgan config test' model.dim=32 model.z_channels=null #data.num_train=3 model.n_codes=16384 model.embed_dim=8 model.perceptual_weight=0.0

# End of File.

