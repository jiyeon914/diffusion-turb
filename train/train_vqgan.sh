#!/bin/bash

#SBATCH -J vq_gan_3d_test               # Job name
#SBATCH -o out.vq_gan_3d_test.%j        # Name of stdout output file (%j expands to %jobId)
#SBATCH -p gpu                           # queue or partiton name
#SBATCH --gres=gpu:1                     # Num Devices

echo
python3 -V
which python3
which pip3
echo
pip3 list | grep pytorch
echo
python3 'train_vqgan.py' run_name='20240611 vqgan test' data.num_train=2 params.batch_size=2

# End of File.

