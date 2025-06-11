import os
import torch
import numpy as np
import pandas as pd
import time
import argparse
import optuna
import random
import gc

import utils
import models

import warnings
warnings.filterwarnings('ignore')

st = time.time()

# ===========================================================================
# ============================ Global parameters ============================
# ===========================================================================

SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

STORAGE_PATH = '/scratch/ssd004/scratch/pimentel/DynamicCOO/outputs/'

NUMBER_OF_TEST_RUNS = 10
TOP_N = 5

# ===========================================================================
# ============================== Model training =============================
# ===========================================================================

def get_count_of_trainable_parameters(
    training_df,
    #
    random_state,
    #
    attention_heads,
    hidden_dimension,
    number_of_hidden_layers,
    dropout_rate,
    global_pooling,
  ):
  
  gc.collect()
  torch.cuda.empty_cache()
  
  torch.manual_seed(random_state)
  torch.cuda.manual_seed_all(random_state)
  np.random.seed(random_state)
  random.seed(random_state)

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  # ===========================================================================
  # =============================== Data loading ==============================
  # ===========================================================================

  if GNN == 'GAT':
    model = models.GAT(
      node_feature_count = EMBEDDING_DIMENSION,
      class_count = np.unique(training_df['label'].values).size,
      attention_heads = attention_heads,
      edge_dimension = AGGREGATION_LEVELS_FEATURE_COUNT[AGGREGATION_LEVEL],
      hidden_dimension = hidden_dimension,
      number_of_hidden_layers = number_of_hidden_layers,
      dropout_rate = dropout_rate,
      global_pooling = global_pooling
    ).to(DEVICE)
    #model = torch.nn.DataParallel(model)
  elif GNN == 'GATv2':
    model = models.GATv2(
      node_feature_count = EMBEDDING_DIMENSION,
      class_count = np.unique(training_df['label'].values).size,
      attention_heads = attention_heads,
      edge_dimension = AGGREGATION_LEVELS_FEATURE_COUNT[AGGREGATION_LEVEL],
      hidden_dimension = hidden_dimension,
      number_of_hidden_layers = number_of_hidden_layers,
      dropout_rate = dropout_rate,
      global_pooling = global_pooling
    ).to(DEVICE)
  else:
    raise ValueError('The selected GNN is not supported. Please choose either GAT or GATv2.')

  # Based on https://www.geeksforgeeks.org/check-the-total-number-of-parameters-in-a-pytorch-model/
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ===========================================================================
# ========================== Information extraction =========================
# ===========================================================================

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_set', required = True, type = str, help = 'The name of the data set.')
  parser.add_argument('--path_to_data_set', required = True, type = str, help = 'The path to the data set\'s training, validation, and testing files.')
  parser.add_argument('--large_language_model', required = True, type = str, help = 'Which large language model to use to construct word graphs.')
  parser.add_argument('--graph_neural_network', required = True, type = str, help = 'Which graph neural network to use during the learning step.')
  parser.add_argument('--surrogate', required = True, type = int, help = 'Whether or not to follow a surrogate approach to construct the graphs, i.e., grouping the nodes based on their indices instead of their identifiers.')
  parser.add_argument('--aggregation_level', required = True, type = int, help = 'Which level of aggregation to apply to the attention coefficients: 0 (no aggregation), 1 (layer-wise aggregation), or 2 (global aggregation).')

  args = parser.parse_args()
  DATASET = args.data_set
  PATH_TO_DATASET = args.path_to_data_set
  LARGE_LANGUAGE_MODEL = args.large_language_model
  GNN = args.graph_neural_network
  SURROGATE = args.surrogate
  AGGREGATION_LEVEL = args.aggregation_level

  if SURROGATE not in [0, 1]:
    raise ValueError('The surrogate parameter must be either 0 (False) or 1 (True).')
  
  if GNN not in ['GAT', 'GATv2']:
    raise ValueError('The selected GNN is not supported. Please choose either GAT or GATv2.')

  if AGGREGATION_LEVEL not in [0, 1, 2]:
    raise ValueError('The aggregation level must be 0 (no aggregation), 1 (layer-wise aggregation), or 2 (global aggregation).')
  
  SURROGATE = bool(SURROGATE)

  _, _, _, TOKENIZER, LLM, MAXIMUM_SEQUENCE_LENGTH, EMBEDDING_OUTPUT_KEY, ATTENTION_OUTPUT_KEY = utils.load_feature_extractor(LARGE_LANGUAGE_MODEL)
  LLM.to(DEVICE)

  training_df = pd.read_csv(os.path.join(PATH_TO_DATASET, 'train.csv'))
  
  EMBEDDING_DIMENSION = LLM.config.hidden_size
  LAYERS = LLM.config.num_hidden_layers
  HEADS = LLM.config.num_attention_heads

  AGGREGATION_LEVELS = ['No_Aggregation', 'Layer-wise_Aggregation', 'Global_Aggregation']
  AGGREGATION_LEVELS_FEATURE_COUNT = [LAYERS * HEADS, LAYERS, 1]
  
  study_name = f'{DATASET}-{GNN}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}-{"Surrogate" if SURROGATE else "Grouped"}-{AGGREGATION_LEVELS[AGGREGATION_LEVEL]}'
  storage = f'sqlite:///../../../pipelines/optuna_studies/{study_name}.db'
    
  sampler = optuna.samplers.TPESampler(seed = SEED)
  study = optuna.create_study(
    direction = 'maximize',
    sampler = sampler,
    study_name = study_name, 
    storage = storage, 
    load_if_exists = True
  )

  trial_features = [
    'number', 'value', 'params_threshold', 'params_attention_heads', 'params_balanced_loss',
    'params_embedding_pooling_operation', 'params_attention_pooling_operation',
    'params_batch_size', 'params_dropout_rate', 'params_early_stopping_patience',
    'params_epochs', 'params_global_pooling', 'params_hidden_dimension',
    'params_learning_rate', 'params_number_of_hidden_layers', 'params_plateau_divider',
    'params_plateau_patience', 'params_weight_decay', 'params_beta_0', 'params_beta_1',
    'params_epsilon', 'user_attrs_epoch', 'user_attrs_training_loss', 'user_attrs_validation_loss'
  ]

  top_N_trials = study.trials_dataframe()[trial_features].sort_values(by = ['value', 'user_attrs_validation_loss', 'user_attrs_training_loss'], ascending = [False, True, True]).head(TOP_N)

  for _, trial in top_N_trials.iterrows():
    
    random_states = [x for x in os.listdir(os.path.join(STORAGE_PATH, f'{DATASET}-{GNN}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}', 'Surrogate' if SURROGATE else 'Grouped', AGGREGATION_LEVELS[AGGREGATION_LEVEL], f'{trial["number"]}')) if os.path.isdir(os.path.join(STORAGE_PATH, f'{DATASET}-{GNN}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}', 'Surrogate' if SURROGATE else 'Grouped', AGGREGATION_LEVELS[AGGREGATION_LEVEL], f'{trial["number"]}', f'{x}'))]
    if len(random_states) < NUMBER_OF_TEST_RUNS:
      continue
    
    print('\n[TRIAL]', trial['number'], '[VALIDATION PERFORMANCE]', trial['value'], '[TRAINING LOSS]', trial['user_attrs_training_loss'], '[VALIDATION LOSS]', trial['user_attrs_validation_loss'], '\n', flush = True)
    print(trial, flush = True)

    trainable_parameters = get_count_of_trainable_parameters(
      training_df,
      #
      random_state = SEED,
      #
      attention_heads = trial['params_attention_heads'],
      hidden_dimension = trial['params_hidden_dimension'],
      number_of_hidden_layers = trial['params_number_of_hidden_layers'],
      dropout_rate = trial['params_dropout_rate'],
      global_pooling = trial['params_global_pooling'],
    )
    print('Count of trainable parameters:', trainable_parameters, flush = True)
    break

print(f'\n[{DATASET}] Elapsed time:', (time.time() - st) / 60, 'minutes.', flush = True)
