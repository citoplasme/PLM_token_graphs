#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH -p a40
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=24GB
#BATCH --job-name=R8+GATv2+BART-large+Grouped+No_Aggregation
#SBATCH --output=R8+GATv2+BART-large+Grouped+No_Aggregation_%j.out

source /opt/lmod/lmod/init/profile
source /scratch/ssd004/scratch/pimentel/DynamicCOO/envfiles/DynamicWindowsPytorch.sh
module use /pkgs/environment-modules/
module load cuda-11.8

cd ..
python main.py --data_set R8 --path_to_data_set /h/pimentel/DynamicCOO/data/with_validation_splits/R8/ --chunking 1 --minimum_threshold 0.65 --maximum_threshold 0.99999 --minimum_batch_size 32 --maximum_batch_size 256 --large_language_model facebook/bart-large --use_f1_score 1 --graph_neural_network GATv2 --surrogate 0 --aggregation_level 0
