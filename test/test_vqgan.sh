#!/bin/bash

#SBATCH -J vq_gan_3d_test_out               # Job name
#SBATCH -o out.vq_gan_3d_test_out.%j        # Name of stdout output file (%j expands to %jobId)
#SBATCH -p gpu                           # queue or partiton name
#SBATCH --gres=gpu:1                     # Num Devices

echo
python3 -V
which python3
which pip3
echo
pip3 list | grep pytorch
echo
python3 'test_vqgan.py' params.batch_size=4 +test_name='20240614 vqgan output' hydra.run.dir='../outputs/test_vqgan/${now:%Y-%m-%d}/${now:%H-%M-%S}'

# End of File.

