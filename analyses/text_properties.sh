#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=normal
#SBATCH -p cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=1GB
#BATCH --job-name=Text_Properties
#SBATCH --output=Text_Properties_%j.out

source /opt/lmod/lmod/init/profile
source /scratch/ssd004/scratch/pimentel/DynamicCOO/envfiles/DynamicWindowsPytorch.sh

python text_properties.py
