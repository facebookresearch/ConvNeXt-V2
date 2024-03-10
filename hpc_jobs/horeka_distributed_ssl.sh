#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --time=48:00:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=luca.rettenberger@kit.edu
#SBATCH --error=%j_error.txt
#SBATCH --output=%j_output.txt
#SBATCH --job-name=hyperparam-sem-seg
#SBATCH --account=hk-project-p0021401

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=12

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# SETUP CUSTOM ENVIORMENT

# remove all modules
module purge
# activate cuda
module load devel/cuda/11.1
# activate conda env
source /home/hk-project-test-dl4pm/hgf_xda8301/miniconda3/etc/profile.d/conda.sh
conda activate sem-segmentation
# move to project dir
cd /home/hk-project-test-dl4pm/hgf_xda8301/ConvNeXt-V2

srun python main_pretrain.py --input_size=640 --mask_ratio=0.5 --patch_size=32 --data_path=/home/hk-project-test-dl4pm/hgf_xda8301/data/sem_segmentation_ssl --output_dir=/home/hk-project-test-dl4pm/hgf_xda8301/ConvNeXt-V2/last_horeka_ressources --pretraining=ssl --warmup_epochs=40
