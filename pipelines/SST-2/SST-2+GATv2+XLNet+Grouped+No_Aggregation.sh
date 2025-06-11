#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=m
#SBATCH --gres=gpu:1
#SBATCH -p a40
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=24GB
#BATCH --job-name=SST-2+GATv2+XLNet+Grouped+No_Aggregation
#SBATCH --output=SST-2+GATv2+XLNet+Grouped+No_Aggregation_%j.out

source /opt/lmod/lmod/init/profile
source /scratch/ssd004/scratch/pimentel/DynamicCOO/envfiles/DynamicWindowsPytorch.sh
module use /pkgs/environment-modules/
module load cuda-11.8

cd ..
python main.py --data_set SST-2 --path_to_data_set /h/pimentel/DynamicCOO/data/with_validation_splits/SST-2/ --chunking 0 --minimum_threshold 0.4 --maximum_threshold 0.95 --minimum_batch_size 32 --maximum_batch_size 256 --large_language_model xlnet/xlnet-base-cased --use_f1_score 0 --graph_neural_network GATv2 --surrogate 0 --aggregation_level 0
