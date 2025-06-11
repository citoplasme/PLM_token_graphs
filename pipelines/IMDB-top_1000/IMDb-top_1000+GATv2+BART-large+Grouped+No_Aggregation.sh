#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH -p a40
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00
#SBATCH --mem=24GB
#BATCH --job-name=IMDb-top_1000+GATv2+BART-large+Grouped+No_Aggregation
#SBATCH --output=IMDb-top_1000+GATv2+BART-large+Grouped+No_Aggregation_%j.out

source /opt/lmod/lmod/init/profile
source /scratch/ssd004/scratch/pimentel/DynamicCOO/envfiles/DynamicWindowsPytorch.sh
module use /pkgs/environment-modules/
module load cuda-11.8

cd ..
python main.py --data_set IMDb-top_1000 --path_to_data_set /h/pimentel/DynamicCOO/data/with_validation_splits/IMDb-top_1000/ --chunking 1 --minimum_threshold 0.9 --maximum_threshold 0.99999 --minimum_batch_size 16 --maximum_batch_size 64 --large_language_model facebook/bart-large --use_f1_score 0 --graph_neural_network GATv2 --surrogate 0 --aggregation_level 0
