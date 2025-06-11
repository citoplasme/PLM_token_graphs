#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH -p a40
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=24GB
#BATCH --job-name=IMDb-top_1000+BART-L+Surrogate+Random_Document
#SBATCH --output=IMDb-top_1000+BART-L+Surrogate+Random_Document_%j.out

source /opt/lmod/lmod/init/profile
source /scratch/ssd004/scratch/pimentel/DynamicCOO/envfiles/DynamicWindowsPytorch.sh
module use /pkgs/environment-modules/
module load cuda-11.8

python random_document_selection.py --data_set IMDb-top_1000 --path_to_data_set /h/pimentel/DynamicCOO/data/with_validation_splits/IMDb-top_1000/ --chunking 0 --large_language_model facebook/bart-large --graph_neural_network GATv2 --surrogate 1 --aggregation_level 0
