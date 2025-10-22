#!/bin/bash

#SBATCH --job-name=run09     # Job name
#SBATCH --output=/data/cluster/logs_DR/run09.out
#SBATCH --error=/data/cluster/logs_DR/run09.err
#SBATCH --nodelist=node09
#SBATCH --exclude=node[17-24]
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=daniel.rose@univie.ac.at
#SBATCH --mail-type=BEGIN,END,FAIL

source /data/sharedXL/software/conda/daniel_rose/mambaforge/etc/profile.d/conda.sh
conda activate molgen
srun python /data/sharedXL/projects/Daniel/MolGen/scripts/training.py --config /data/sharedXL/projects/Daniel/MolGen/scripts/config.yaml