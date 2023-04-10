#!/bin/bash -e
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --time=2-00:00:00
#SBATCH --job-name=jobName
#SBATCH --output=allout/slurm_%j.out
#SBATCH --error=allout/slurm_%j.err
#SBATCH --chdir=/scratch/sca321/research/convnext3/ConvNeXt-V2
#SBATCH --mem=72G
#SBATCH --gres=gpu:a100:4

module purge

singularity exec --nv \
        --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
        --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
	    --overlay /scratch/sca321/research/convnext3/conda/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python -m torch.distributed.launch --nproc_per_node=4 main_pretrain_clipteacher.py --epochs 200 --num_workers 6 --clip_model ViT-B/32"
