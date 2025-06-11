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
# ============================== Model training =============================
# ===========================================================================

def get_count_of_trainable_parameters(
    training_df,
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

  _, model, _ = utils.load_feature_extractor(LARGE_LANGUAGE_MODEL, num_labels = np.unique(training_df['label'].values).size)
  model.to(DEVICE)

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
  
  args = parser.parse_args()
  DATASET = args.data_set
  PATH_TO_DATASET = args.path_to_data_set
  LARGE_LANGUAGE_MODEL = args.large_language_model
  
  training_df = pd.read_csv(os.path.join(PATH_TO_DATASET, 'train.csv'))
  
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

    trainable_parameters = get_count_of_trainable_parameters(
      training_df,
      #
      random_state = SEED
    )
    print('Count of trainable parameters:', trainable_parameters, flush = True)
    break

print(f'\n[{DATASET}] Elapsed time:', (time.time() - st) / 60, 'minutes.', flush = True)
