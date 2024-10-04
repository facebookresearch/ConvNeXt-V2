#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luca.rettenberger@kit.edu
#SBATCH --error=%j_error.txt
#SBATCH --output=%j_output.txt
#SBATCH --job-name=hyperparam-sem-seg
#SBATCH --account=hk-project-p0021401

# remove all modules
module purge
# activate cuda
module load devel/cuda/11.8
# activate conda env
source /home/hk-project-test-dl4pm/hgf_xda8301/miniconda3/etc/profile.d/conda.sh
conda activate sem-segmentation
# move to project dir
cd /home/hk-project-test-dl4pm/hgf_xda8301/ConvNeXt-V2

python main_pretrain.py --input_size=640 --mask_ratio=0.5 --patch_size=32 --data_path=/home/hk-project-test-dl4pm/hgf_xda8301/data/insect_ssl --output_dir=/home/hk-project-test-dl4pm/hgf_xda8301/ConvNeXt-V2/insect_ssl_small_single_gpu --warmup_epochs=40

