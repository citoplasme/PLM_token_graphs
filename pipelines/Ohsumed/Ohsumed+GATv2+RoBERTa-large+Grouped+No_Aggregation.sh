#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH -p a40
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=24GB
#BATCH --job-name=Ohsumed+GATv2+RoBERTa-large+Grouped+No_Aggregation
#SBATCH --output=Ohsumed+GATv2+RoBERTa-large+Grouped+No_Aggregation_%j.out

source /opt/lmod/lmod/init/profile
source /scratch/ssd004/scratch/pimentel/DynamicCOO/envfiles/DynamicWindowsPytorch.sh
module use /pkgs/environment-modules/
module load cuda-11.8

cd ..
python main.py --data_set Ohsumed --path_to_data_set /h/pimentel/DynamicCOO/data/with_validation_splits/Ohsumed/ --chunking 1 --minimum_threshold 0.8 --maximum_threshold 0.99999 --minimum_batch_size 32 --maximum_batch_size 128 --large_language_model FacebookAI/roberta-large --use_f1_score 1 --graph_neural_network GATv2 --surrogate 0 --aggregation_level 0
