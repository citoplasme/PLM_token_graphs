#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH -p a40
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=24GB
#BATCH --job-name=R8+BERT
#SBATCH --output=R8+BERT_%j.out

source /opt/lmod/lmod/init/profile
source /scratch/ssd004/scratch/pimentel/DynamicCOO/envfiles/DynamicWindowsPytorch.sh
module use /pkgs/environment-modules/
module load cuda-11.8

cd ..
python main.py --data_set R8 --path_to_data_set /h/pimentel/DynamicCOO/data/with_validation_splits/R8/ --minimum_batch_size 8 --maximum_batch_size 32 --large_language_model google-bert/bert-base-uncased --use_f1_score 1
