#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=m
#SBATCH --gres=gpu:1
#SBATCH -p a40
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=24GB
#BATCH --job-name=Limitations
#SBATCH --output=Limitations_%j.out

source /opt/lmod/lmod/init/profile
source /scratch/ssd004/scratch/pimentel/DynamicCOO/envfiles/DynamicWindowsPytorch.sh
module use /pkgs/environment-modules/
module load cuda-11.8

python main.py