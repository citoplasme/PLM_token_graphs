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
import construct_graphs

import warnings
warnings.filterwarnings('ignore')

st = time.time()

# ===========================================================================
# ============================ Global parameters ============================
# ===========================================================================

SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

STORAGE_PATH = '/scratch/ssd004/scratch/pimentel/DynamicCOO/sliding_window_outputs/'

NUMBER_OF_TEST_RUNS = 10
TOP_N = 5

# ===========================================================================
# ============================== Model training =============================
# ===========================================================================

def get_values_per_document_distribution(
    training_df,
    validation_df,
    testing_df,
    #
    random_state,
    #
    chunk_size,
    left_stride,
    right_stride,
    co_occurrence_pooling_operation,
    embedding_pooling_operation,
    window_size,
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

  os.makedirs(os.path.join('..', 'data', f'{LARGE_LANGUAGE_MODEL.replace("/", "-")}-{"Surrogate" if SURROGATE else "Grouped"}-Sliding_Windows'), exist_ok = True)

  l = list()
  for i, row in training_df.iterrows():
    graph = construct_graphs.construct_PyG_graph_from_LLM_using_sliding_windows(
      text = row['text'],
      label = row['label'],
      split = 'train',
      index = i,
      #
      chunk_size = chunk_size,
      left_stride = left_stride,
      right_stride = right_stride,
      #
      surrogate = SURROGATE,
      embedding_pooling = embedding_pooling_operation,
      window_size = window_size,
      co_occurrence_pooling = co_occurrence_pooling_operation,
      #
      llm = LLM,
      tokenizer = TOKENIZER,
      embedding_output_key = EMBEDDING_OUTPUT_KEY,
      maximum_chunk_size = MAXIMUM_SEQUENCE_LENGTH,
      #
      device = DEVICE
    )
    
    nodes = graph.num_nodes
    total_values_for_nodes = nodes * EMBEDDING_DIMENSION
    
    edges = graph.num_edges
    total_values_for_edges = edges * 1
    
    total_values_for_graph = total_values_for_nodes + total_values_for_edges
    
    split = graph['split']
    index = graph['identifier']
    
    l.append((split, index.item(), nodes, total_values_for_nodes, edges, total_values_for_edges, total_values_for_graph))

  for i, row in validation_df.iterrows():
    graph = construct_graphs.construct_PyG_graph_from_LLM_using_sliding_windows(
      text = row['text'],
      label = row['label'],
      split = 'validation',
      index = i,
      #
      chunk_size = chunk_size,
      left_stride = left_stride,
      right_stride = right_stride,
      #
      surrogate = SURROGATE,
      embedding_pooling = embedding_pooling_operation,
      window_size = window_size,
      co_occurrence_pooling = co_occurrence_pooling_operation,
      #
      llm = LLM,
      tokenizer = TOKENIZER,
      embedding_output_key = EMBEDDING_OUTPUT_KEY,
      maximum_chunk_size = MAXIMUM_SEQUENCE_LENGTH,
      #
      device = DEVICE
    )

    nodes = graph.num_nodes
    total_values_for_nodes = nodes * EMBEDDING_DIMENSION
    
    edges = graph.num_edges
    total_values_for_edges = edges * 1
    
    total_values_for_graph = total_values_for_nodes + total_values_for_edges
    
    split = graph['split']
    index = graph['identifier']
    
    l.append((split, index.item(), nodes, total_values_for_nodes, edges, total_values_for_edges, total_values_for_graph))

  for i, row in testing_df.iterrows():
    graph = construct_graphs.construct_PyG_graph_from_LLM_using_sliding_windows(
      text = row['text'],
      label = row['label'],
      split = 'test',
      index = i,
      #
      chunk_size = chunk_size,
      left_stride = left_stride,
      right_stride = right_stride,
      #
      surrogate = SURROGATE,
      embedding_pooling = embedding_pooling_operation,
      window_size = window_size,
      co_occurrence_pooling = co_occurrence_pooling_operation,
      #
      llm = LLM,
      tokenizer = TOKENIZER,
      embedding_output_key = EMBEDDING_OUTPUT_KEY,
      maximum_chunk_size = MAXIMUM_SEQUENCE_LENGTH,
      #
      device = DEVICE
    )

    nodes = graph.num_nodes
    total_values_for_nodes = nodes * EMBEDDING_DIMENSION
    
    edges = graph.num_edges
    total_values_for_edges = edges * 1
    
    total_values_for_graph = total_values_for_nodes + total_values_for_edges
    
    split = graph['split']
    index = graph['identifier']
    
    l.append((split, index.item(), nodes, total_values_for_nodes, edges, total_values_for_edges, total_values_for_graph))
  
  pd.DataFrame(l, columns = ['split', 'index', 'nodes', 'total_values_for_nodes', 'edges', 'total_values_for_edges', 'total_values_for_graph']) \
    .to_csv(os.path.join('..', 'data', f'{LARGE_LANGUAGE_MODEL.replace("/", "-")}-{"Surrogate" if SURROGATE else "Grouped"}-Sliding_Windows', f'{DATASET}.csv'), index = False)

# ===========================================================================
# ========================== Information extraction =========================
# ===========================================================================

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_set', required = True, type = str, help = 'The name of the data set.')
  parser.add_argument('--path_to_data_set', required = True, type = str, help = 'The path to the data set\'s training, validation, and testing files.')
  parser.add_argument('--chunking', required = True, type = int, help = 'Whether or not to chunk the input text depending on the maximum sequence length accepted by the model and the properties of the text.')
  parser.add_argument('--large_language_model', required = True, type = str, help = 'Which large language model to use to construct word graphs.')
  parser.add_argument('--graph_neural_network', required = True, type = str, help = 'Which graph neural network to use during the learning step.')
  parser.add_argument('--surrogate', required = True, type = int, help = 'Whether or not to follow a surrogate approach to construct the graphs, i.e., grouping the nodes based on their indices instead of their identifiers.')

  args = parser.parse_args()
  DATASET = args.data_set
  PATH_TO_DATASET = args.path_to_data_set
  CHUNKING = args.chunking
  LARGE_LANGUAGE_MODEL = args.large_language_model
  GNN = args.graph_neural_network
  SURROGATE = args.surrogate

  if CHUNKING not in [0, 1]:
    raise ValueError('The chunking parameter must be either 0 (False) or 1 (True).')

  if SURROGATE not in [0, 1]:
    raise ValueError('The surrogate parameter must be either 0 (False) or 1 (True).')

  if GNN not in ['GAT', 'GATv2']:
    raise ValueError('The selected GNN is not supported. Please choose either GAT or GATv2.')
  
  CHUNKING = bool(CHUNKING)
  SURROGATE = bool(SURROGATE)

  _, _, _, TOKENIZER, LLM, MAXIMUM_SEQUENCE_LENGTH, EMBEDDING_OUTPUT_KEY, ATTENTION_OUTPUT_KEY = utils.load_feature_extractor(LARGE_LANGUAGE_MODEL)
  LLM.to(DEVICE)

  training_df = pd.read_csv(os.path.join(PATH_TO_DATASET, 'train.csv'))
  validation_df = pd.read_csv(os.path.join(PATH_TO_DATASET, 'validation.csv'))  
  testing_df = pd.read_csv(os.path.join(PATH_TO_DATASET, 'test.csv'))

  EMBEDDING_DIMENSION = LLM.config.hidden_size

  study_name = f'{DATASET}-{GNN}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}-{"Surrogate" if SURROGATE else "Grouped"}-Sliding_Windows'
  storage = f'sqlite:///../../../sliding_windows/optuna_studies/{study_name}.db'
    
  sampler = optuna.samplers.TPESampler(seed = SEED)
  study = optuna.create_study(
    direction = 'maximize',
    sampler = sampler,
    study_name = study_name, 
    storage = storage, 
    load_if_exists = True
  )

  trial_features = [
    'number', 'value', 'params_window_size', 'params_attention_heads', 'params_balanced_loss',
    'params_embedding_pooling_operation', 'params_co_occurrence_pooling_operation',
    'params_batch_size', 'params_dropout_rate', 'params_early_stopping_patience',
    'params_epochs', 'params_global_pooling', 'params_hidden_dimension',
    'params_learning_rate', 'params_number_of_hidden_layers', 'params_plateau_divider',
    'params_plateau_patience', 'params_weight_decay', 'params_beta_0', 'params_beta_1',
    'params_epsilon', 'user_attrs_epoch', 'user_attrs_training_loss', 'user_attrs_validation_loss'
  ]

  if CHUNKING:
    trial_features += ['params_left_stride', 'params_right_stride'] #['params_chunk_size', 'params_left_stride', 'params_right_stride']

  top_N_trials = study.trials_dataframe()[trial_features].sort_values(by = ['value', 'user_attrs_validation_loss', 'user_attrs_training_loss'], ascending = [False, True, True]).head(TOP_N)

  for _, trial in top_N_trials.iterrows():
    
    random_states = [x for x in os.listdir(os.path.join(STORAGE_PATH, f'{DATASET}-{GNN}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}', 'Surrogate' if SURROGATE else 'Grouped', 'Sliding_Windows', f'{trial["number"]}')) if os.path.isdir(os.path.join(STORAGE_PATH, f'{DATASET}-{GNN}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}', 'Surrogate' if SURROGATE else 'Grouped', 'Sliding_Windows', f'{trial["number"]}', f'{x}'))]
    if len(random_states) < NUMBER_OF_TEST_RUNS:
      continue
    
    print('\n[TRIAL]', trial['number'], '[VALIDATION PERFORMANCE]', trial['value'], '[TRAINING LOSS]', trial['user_attrs_training_loss'], '[VALIDATION LOSS]', trial['user_attrs_validation_loss'], '\n', flush = True)
    print(trial, flush = True)

    get_values_per_document_distribution(
      training_df,
      validation_df,
      testing_df,
      #
      random_state = SEED,
      #
      chunk_size = MAXIMUM_SEQUENCE_LENGTH, # trial['params_chunk_size'] if CHUNKING else MAXIMUM_SEQUENCE_LENGTH,
      left_stride = trial['params_left_stride'] if CHUNKING else 0,
      right_stride = trial['params_right_stride'] if CHUNKING else 0,
      co_occurrence_pooling_operation = trial['params_co_occurrence_pooling_operation'],
      embedding_pooling_operation = trial['params_embedding_pooling_operation'],
      window_size = trial['params_window_size'],
    )
    break

print(f'\n[{DATASET}] Elapsed time:', (time.time() - st) / 60, 'minutes.', flush = True)
