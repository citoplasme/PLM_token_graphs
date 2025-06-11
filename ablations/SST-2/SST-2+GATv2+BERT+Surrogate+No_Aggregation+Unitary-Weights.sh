#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=m
#SBATCH --gres=gpu:1
#SBATCH -p a40
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=24GB
#BATCH --job-name=SST-2+GATv2+BERT+Surrogate+No_Aggregation+Unitary_Weights
#SBATCH --output=SST-2+GATv2+BERT+Surrogate+No_Aggregation+Unitary_Weights_%j.out

source /opt/lmod/lmod/init/profile
source /scratch/ssd004/scratch/pimentel/DynamicCOO/envfiles/DynamicWindowsPytorch.sh
module use /pkgs/environment-modules/
module load cuda-11.8

cd ..
python main.py --data_set SST-2 --path_to_data_set /h/pimentel/DynamicCOO/data/with_validation_splits/SST-2/ --chunking 0 --minimum_threshold 0.4 --maximum_threshold 0.95 --minimum_batch_size 32 --maximum_batch_size 256 --large_language_model google-bert/bert-base-uncased --use_f1_score 0 --graph_neural_network GATv2 --surrogate 1 --aggregation_level 0 --ablation_operation 2 --node_ablation_noise_level 0.0 --node_ablation_feature_noise_level 0.0
