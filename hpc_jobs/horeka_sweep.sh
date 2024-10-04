#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luca.rettenberger@kit.edu
#SBATCH --error=%j_error.txt
#SBATCH --output=%j_output.txt
#SBATCH --job-name=hyperparam-sem-seg

export SWEEPID="kit-iai-ibcs-dl/sem-segmentation/rfijcjry"

# aendern?
NUM_COUNTS=1

# remove all modules
module purge

# activate cuda
module load devel/cuda/11.1

# activate conda env
source /home/hk-project-test-dl4pm/hgf_xda8301/miniconda3/etc/profile.d/conda.sh
conda activate sem-segmentation


cd /home/hk-project-test-dl4pm/hgf_xda8301/ConvNeXt-V2


# start train
CUDA_VISIBLE_DEVICES=0 wandb agent $SWEEPID &
CUDA_VISIBLE_DEVICES=1 wandb agent $SWEEPID &
CUDA_VISIBLE_DEVICES=2 wandb agent $SWEEPID &
CUDA_VISIBLE_DEVICES=3 wandb agent $SWEEPID

wait < <(jobs -p)
