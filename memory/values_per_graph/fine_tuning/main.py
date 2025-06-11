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
# ================================ Data Set =================================
# ===========================================================================

# Based on https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CustomDataset(torch.utils.data.Dataset):
  
  def __init__(self, df, tokenizer, max_tokens, split):
    self.df = df
    self.tokenizer = tokenizer
    self.max_tokens = max_tokens
    self.split = split
        
  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, idx):

    text = self.df.iloc[idx]['text']
    label = self.df.iloc[idx]['label']

    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens = True,
      max_length = self.max_tokens, # https://stackoverflow.com/questions/58636587/how-to-use-bert-for-long-text-classification
      #padding = 'max_length', # Removed for this experiment 
      truncation = True,
      return_attention_mask = True,
      return_tensors = 'pt',
    )
    return {
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask' : encoding['attention_mask'].flatten(),
      'y' : torch.tensor(label, dtype = torch.long),
      'split' : self.split,
      'index' : idx
    }

# ===========================================================================
# ============================== Model training =============================
# ===========================================================================

def get_values_per_document_distribution(
    training_df,
    validation_df,
    testing_df,
    #
    random_state,
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

  TOKENIZER, model, MAXIMUM_SEQUENCE_LENGTH = utils.load_feature_extractor(LARGE_LANGUAGE_MODEL, num_labels = np.unique(training_df['label'].values).size)
  model.to(DEVICE)

  EMBEDDING_DIMENSION = model.config.hidden_size
  LAYERS = model.config.num_hidden_layers
  HEADS = model.config.num_attention_heads

  training_dataset = CustomDataset(training_df, TOKENIZER, MAXIMUM_SEQUENCE_LENGTH, 'train')
  validation_dataset = CustomDataset(validation_df, TOKENIZER, MAXIMUM_SEQUENCE_LENGTH, 'validation')
  testing_dataset = CustomDataset(testing_df, TOKENIZER, MAXIMUM_SEQUENCE_LENGTH, 'test')

  os.makedirs(os.path.join('..', 'data', 'fine_tuning'), exist_ok = True)

  l = list()
  for item in training_dataset:
    tokens = item['input_ids'].size(0)
    total_values_for_tokens = tokens * EMBEDDING_DIMENSION
    
    attention_coefficients = tokens * tokens
    total_values_attention_coefficients = attention_coefficients * LAYERS * HEADS
    
    total_values = total_values_for_tokens + total_values_attention_coefficients
    
    split = item['split']
    index = item['index']
    
    l.append((split, index, tokens, total_values_for_tokens, attention_coefficients, total_values_attention_coefficients, total_values))
  
  for item in validation_dataset:
    tokens = item['input_ids'].size(0)
    total_values_for_tokens = tokens * EMBEDDING_DIMENSION
    
    attention_coefficients = tokens * tokens
    total_values_attention_coefficients = attention_coefficients * LAYERS * HEADS
    
    total_values = total_values_for_tokens + total_values_attention_coefficients
    
    split = item['split']
    index = item['index']
    
    l.append((split, index, tokens, total_values_for_tokens, attention_coefficients, total_values_attention_coefficients, total_values))
  
  for item in testing_dataset:
    tokens = item['input_ids'].size(0)
    total_values_for_tokens = tokens * EMBEDDING_DIMENSION
    
    attention_coefficients = tokens * tokens
    total_values_attention_coefficients = attention_coefficients * LAYERS * HEADS
    
    total_values = total_values_for_tokens + total_values_attention_coefficients
    
    split = item['split']
    index = item['index']
    
    l.append((split, index, tokens, total_values_for_tokens, attention_coefficients, total_values_attention_coefficients, total_values))
  
  pd.DataFrame(l, columns = ['split', 'index', 'nodes', 'total_values_for_nodes', 'edges', 'total_values_for_edges', 'total_values_for_graph']) \
    .to_csv(os.path.join('..', 'data', 'fine_tuning', f'{DATASET}.csv'), index = False)
    
# ===========================================================================
# ========================== Information extraction =========================
# ===========================================================================

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_set', required = True, type = str, help = 'The name of the data set.')
  parser.add_argument('--path_to_data_set', required = True, type = str, help = 'The path to the data set\'s training, validation, and testing files.')
  parser.add_argument('--large_language_model', required = True, type = str, help = 'Which large language model to use to construct word graphs.')
  
  args = parser.parse_args()
  DATASET = args.data_set
  PATH_TO_DATASET = args.path_to_data_set
  LARGE_LANGUAGE_MODEL = args.large_language_model
    
  training_df = pd.read_csv(os.path.join(PATH_TO_DATASET, 'train.csv'))
  validation_df = pd.read_csv(os.path.join(PATH_TO_DATASET, 'validation.csv'))  
  testing_df = pd.read_csv(os.path.join(PATH_TO_DATASET, 'test.csv'))

  study_name = f'{DATASET}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}'
  storage = f'sqlite:///../../../baselines/optuna_studies/{study_name}.db'
    
  sampler = optuna.samplers.TPESampler(seed = SEED)
  study = optuna.create_study(
    direction = 'maximize',
    sampler = sampler,
    study_name = study_name, 
    storage = storage, 
    load_if_exists = True
  )

  trial_features = [
    'number', 'value', 'params_balanced_loss', 'params_batch_size',
    'params_early_stopping_patience', 'params_epochs', 'params_learning_rate', 'params_plateau_divider',
    'params_plateau_patience', 'params_weight_decay', 'params_beta_0', 'params_beta_1',
    'params_epsilon', 'user_attrs_epoch', 'user_attrs_training_loss', 'user_attrs_validation_loss'
  ]

  top_N_trials = study.trials_dataframe()[trial_features].sort_values(by = ['value', 'user_attrs_validation_loss', 'user_attrs_training_loss'], ascending = [False, True, True]).head(TOP_N)

  for _, trial in top_N_trials.iterrows():
    
    random_states = [x for x in os.listdir(os.path.join(STORAGE_PATH, f'{DATASET}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}', f'{trial["number"]}')) if os.path.isdir(os.path.join(STORAGE_PATH, f'{DATASET}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}', f'{trial["number"]}', f'{x}'))]
    if len(random_states) < NUMBER_OF_TEST_RUNS:
      continue
    
    print('\n[TRIAL]', trial['number'], '[VALIDATION PERFORMANCE]', trial['value'], '[TRAINING LOSS]', trial['user_attrs_training_loss'], '[VALIDATION LOSS]', trial['user_attrs_validation_loss'], '\n', flush = True)
    print(trial, flush = True)

    get_values_per_document_distribution(
      training_df,
      validation_df,
      testing_df,
      #
      random_state = SEED
    )
    break

print(f'\n[{DATASET}] Elapsed time:', (time.time() - st) / 60, 'minutes.', flush = True)
