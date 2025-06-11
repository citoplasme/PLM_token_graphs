#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH -p a40
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=24GB
#BATCH --job-name=Ohsumed+BERT
#SBATCH --output=Ohsumed+BERT_%j.out

source /opt/lmod/lmod/init/profile
source /scratch/ssd004/scratch/pimentel/DynamicCOO/envfiles/DynamicWindowsPytorch.sh
module use /pkgs/environment-modules/
module load cuda-11.8

cd ..
python main.py --data_set Ohsumed --path_to_data_set /h/pimentel/DynamicCOO/data/with_validation_splits/Ohsumed/ --large_language_model google-bert/bert-base-uncased
