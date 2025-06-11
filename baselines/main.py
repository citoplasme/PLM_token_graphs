import os
import torch
import numpy as np
import pandas as pd
import time
import argparse
import optuna
import sklearn.model_selection
import sklearn.metrics
import sklearn.utils
import statistics
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

TRIALS = 250
NUMBER_OF_TEST_RUNS = 10
TOP_N = 5

CHECKPOINT_PATH = f'/checkpoint/pimentel/{os.environ["SLURM_JOB_ID"]}/'

# ===========================================================================
# ============================= Auxiliary methods ===========================
# ===========================================================================

def callback(study, trial):
  if (study.trials_dataframe()['value'] >= 0.0).sum() >= TRIALS:
    study.stop()

def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

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
      padding = 'max_length',
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

def train_and_predict(
    training_df,
    validation_df,
    testing_df,
    #
    random_state,
    #
    batch_size,
    learning_rate,
    weight_decay,
    beta_0,
    beta_1,
    epsilon,
    balanced_loss,
    epochs,
    early_stopping_patience,
    plateau_patience,
    plateau_divider,
    #
    evaluate_test = False,
    trial_number = 0
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

  try:

    TOKENIZER, model, MAXIMUM_SEQUENCE_LENGTH = utils.load_feature_extractor(LARGE_LANGUAGE_MODEL, num_labels = np.unique(training_df['label'].values).size)
    model.to(DEVICE)

    training_dataset = CustomDataset(training_df, TOKENIZER, MAXIMUM_SEQUENCE_LENGTH, 'train')
    validation_dataset = CustomDataset(validation_df, TOKENIZER, MAXIMUM_SEQUENCE_LENGTH, 'validation')
    testing_dataset = CustomDataset(testing_df, TOKENIZER, MAXIMUM_SEQUENCE_LENGTH, 'test')
    
    # ===========================================================================
    # ============================== Batchification =============================
    # ===========================================================================

    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())

    training_batches = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, num_workers = 0, worker_init_fn = seed_worker, generator = generator, shuffle = True)
    validation_batches = torch.utils.data.DataLoader(validation_dataset, batch_size = batch_size, num_workers = 0, worker_init_fn = seed_worker, generator = generator, shuffle = True)
    testing_batches = torch.utils.data.DataLoader(testing_dataset, batch_size = batch_size, num_workers = 0, worker_init_fn = seed_worker, generator = generator, shuffle = True)

    # ===========================================================================
    # ================================= Training ================================
    # ===========================================================================

    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, betas = (beta_0, beta_1), eps = epsilon, weight_decay = weight_decay)

    # https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    if balanced_loss:
      CLASS_WEIGHTS = torch.tensor(
        sklearn.utils.class_weight.compute_class_weight(
          class_weight = 'balanced', 
          classes = np.unique(training_df['label'].values), 
          y = training_df['label'].values, #[graph.y.item() for graph in training_graphs]
        ), dtype = torch.float).to(DEVICE)
    else:
      CLASS_WEIGHTS = None
    criterion = torch.nn.CrossEntropyLoss(weight = CLASS_WEIGHTS)

    best_validation_performance = 0.0  
    best_validation_performance_loss = float('inf')
    best_training_performance_loss = float('inf')
    best_validation_performance_epoch = 0
    best_validation_loss = float('inf')
    best_validation_labels = list()
    best_validation_predictions = list()
    best_validation_indices = list()
    plateau_counter = 0
    early_stopping_counter = 0

    dynamic_learning_rate = learning_rate

    epoch_runtimes = list()

    for epoch in range(epochs):
      
      model.train()
      total_loss = 0

      epoch_start_time = time.time()

      for batch in training_batches:
        
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        y = batch['y'].to(DEVICE)
        
        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask = attention_mask, labels = y)
        
        loss = criterion(outputs.logits, y)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

      epoch_end_time = time.time()
      epoch_runtimes.append(epoch_end_time - epoch_start_time)

      average_training_loss = total_loss / len(training_batches)
      #print(f'Average training loss for epoch {epoch + 1}: {avg_train_loss:.4f}', flush = True)

      # ===========================================================================
      # ================================ Validating ===============================
      # ===========================================================================

      model.eval()
      total_validation_loss = 0
      validation_predictions = []
      validation_labels = []
      validation_indices = []

      with torch.no_grad():
        for batch in validation_batches:
          
          input_ids = batch['input_ids'].to(DEVICE)
          attention_mask = batch['attention_mask'].to(DEVICE)
          y = batch['y'].to(DEVICE)
          index = batch['index'].to(DEVICE)

          outputs = model(input_ids, attention_mask = attention_mask, labels = y)
          predictions = outputs.logits.argmax(dim = 1)

          loss = criterion(outputs.logits, y)
          total_validation_loss += loss.item()

          validation_predictions.extend(predictions.cpu().numpy())
          validation_labels.extend(y.cpu().numpy())
          validation_indices.extend(index.cpu().numpy())

      average_validation_loss = total_validation_loss / len(validation_batches)
      validation_performance = sklearn.metrics.f1_score(validation_labels, validation_predictions, average = 'macro') if F1_SCORE else sklearn.metrics.accuracy_score(validation_labels, validation_predictions)

      #print('[Val]', validation_performance, average_validation_loss, flush = True)

      if (validation_performance > best_validation_performance) or ((validation_performance == best_validation_performance) and (average_validation_loss < best_validation_performance_loss)):
        best_validation_performance = validation_performance
        best_validation_performance_loss = average_validation_loss
        best_training_performance_loss = average_training_loss
        best_validation_performance_epoch = epoch + 1
        best_validation_labels = validation_labels
        best_validation_predictions = validation_predictions
        best_validation_indices = validation_indices
        torch.save({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'optimizer' : optimizer.state_dict(),
        }, os.path.join(CHECKPOINT_PATH, f'best-model-{DATASET}.pth.tar'))
      
      # Update loss on plateau and early stopping
      if average_validation_loss < best_validation_loss:
        best_validation_loss = average_validation_loss
        plateau_counter = 0
        early_stopping_counter = 0
      else:
        plateau_counter += 1
        early_stopping_counter += 1
        if early_stopping_counter > early_stopping_patience:
          break
        if plateau_counter >= plateau_patience:
          dynamic_learning_rate /= plateau_divider
          for param_group in optimizer.param_groups:
            param_group['lr'] /= plateau_divider
          plateau_counter = 0
    
    #print('[Best-val]', best_validation_performance, best_validation_performance_loss, flush = True)
    
    if not evaluate_test:

      # Remove model from GPU
      del model
      
      return best_validation_performance, best_validation_performance_loss, best_training_performance_loss, best_validation_performance_epoch
    else:
      # ===========================================================================
      # ================================== Testing ================================
      # ===========================================================================

      if os.path.exists(os.path.join(CHECKPOINT_PATH, f'best-model-{DATASET}.pth.tar')):
        checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, f'best-model-{DATASET}.pth.tar'))
        #epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

      model.eval()
      test_predictions = []
      test_labels = []
      #test_probability = []
      test_indices = []

      evaluation_runtime = 0

      with torch.no_grad():
        for batch in testing_batches:

          evaluation_start_time = time.time()

          input_ids = batch['input_ids'].to(DEVICE)
          attention_mask = batch['attention_mask'].to(DEVICE)
          y = batch['y'].to(DEVICE)
          index = batch['index'].to(DEVICE)
          
          outputs = model(input_ids, attention_mask = attention_mask, labels = y)
          predictions = outputs.logits.argmax(dim = 1)

          evaluation_runtime = evaluation_runtime + (time.time() - evaluation_start_time)

          test_predictions.extend(predictions.cpu().numpy())
          test_labels.extend(y.cpu().numpy())
          #test_probability.extend(outputs.cpu().numpy())
          test_indices.extend(index.cpu().numpy())
      
      # Average evaluation time per instance
      average_evaluation_runtime = evaluation_runtime / testing_dataset.__len__()

      # Remove model from GPU
      del model

      #with open(f'{DATASET}-GAT-{LARGE_LANGUAGE_MODEL.replace("/", "-")}.json', 'w') as f:
      #  json.dump(sklearn.metrics.classification_report(test_labels, test_predictions, output_dict = True), f)

      pd.DataFrame({
        'test_performance' : [sklearn.metrics.f1_score(test_labels, test_predictions, average = 'macro') if F1_SCORE else sklearn.metrics.accuracy_score(test_labels, test_predictions)],
        'validation_performance' : [best_validation_performance],
        'test_average_evaluation_runtime' : [average_evaluation_runtime],
        'checkpoint_epoch' : [best_validation_performance_epoch],
        'random_state' : [random_state]
      }).to_csv(
        os.path.join(STORAGE_PATH, f'{DATASET}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}', f'{trial_number}', 'metrics.csv'),
        mode = 'a',
        header = not os.path.exists(os.path.join(STORAGE_PATH, f'{DATASET}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}', f'{trial_number}', 'metrics.csv')),
        index = False
      )

      os.makedirs(os.path.join(STORAGE_PATH, f'{DATASET}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}', f'{trial_number}', f'{random_state}'), exist_ok = True)

      pd.DataFrame({
        'real' : test_labels + best_validation_labels,
        'prediction' : test_predictions + best_validation_predictions,
        'index' : test_indices + best_validation_indices,
        'split' : ['test'] * len(test_labels) + ['validation'] * len(best_validation_labels)
      }).to_csv(os.path.join(STORAGE_PATH, f'{DATASET}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}', f'{trial_number}', f'{random_state}', 'predictions.csv'), index = False)

      pd.DataFrame({
        'epoch_runtime' : epoch_runtimes,
      }).to_csv(os.path.join(STORAGE_PATH, f'{DATASET}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}', f'{trial_number}', f'{random_state}', 'runtimes.csv'), index = False)

      #print('Real:        ', test_labels, flush = True)
      #print('Predicted:   ', test_predictions, flush = True)
      #print('Probability: ', test_probability, flush = True)

      return sklearn.metrics.f1_score(test_labels, test_predictions, average = 'macro') if F1_SCORE else sklearn.metrics.accuracy_score(test_labels, test_predictions), best_validation_performance, average_evaluation_runtime, best_validation_performance_epoch
  except Exception as e:
    print(e, flush = True)
    return -1.0, float('inf'), float('inf'), 0

def objective_function(trial):

  #
  batch_size = trial.suggest_int('batch_size', MINIMUM_BATCH_SIZE, MAXIMUM_BATCH_SIZE)
  learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4, log = True)
  weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log = True)
  beta_0 = trial.suggest_float('beta_0', 0.8, 0.9, log = True)
  beta_1 = trial.suggest_float('beta_1', 0.98, 0.999, log = True)
  epsilon = trial.suggest_float('epsilon', 1e-8, 1e-4, log = True)
  balanced_loss = trial.suggest_categorical('balanced_loss', [True, False])
  epochs = trial.suggest_int('epochs', 3, 15)
  early_stopping_patience = trial.suggest_int('early_stopping_patience', 2, 5)
  plateau_patience = trial.suggest_int('plateau_patience', 2, 5)
  plateau_divider = trial.suggest_int('plateau_divider', 2, 10)

  validation_performance, validation_loss, training_loss, epoch = train_and_predict(
    training_df,
    validation_df,
    testing_df,
    #
    random_state = SEED,
    #
    batch_size = batch_size,
    learning_rate = learning_rate,
    weight_decay = weight_decay,
    beta_0 = beta_0,
    beta_1 = beta_1,
    epsilon = epsilon,
    balanced_loss = balanced_loss,
    epochs = epochs,
    early_stopping_patience = early_stopping_patience,
    plateau_patience = plateau_patience,
    plateau_divider = plateau_divider
  )

  trial.set_user_attr('validation_loss', validation_loss)
  trial.set_user_attr('training_loss', training_loss)
  trial.set_user_attr('epoch', epoch)

  return validation_performance

# ===========================================================================
# ========================== Information extraction =========================
# ===========================================================================

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_set', required = True, type = str, help = 'The name of the data set.')
  parser.add_argument('--path_to_data_set', required = True, type = str, help = 'The path to the data set\'s training, validation, and testing files.')
  parser.add_argument('--minimum_batch_size', required = True, type = float, help = 'The minimum value for batch size used during hyper-parameter tuning.')
  parser.add_argument('--maximum_batch_size', required = True, type = float, help = 'The maximum value for batch size used during hyper-parameter tuning.')
  parser.add_argument('--large_language_model', required = True, type = str, help = 'Which large language model to use to construct word graphs.')
  parser.add_argument('--use_f1_score', required = True, type = int, help = 'Whether or not to evaluate the models using macro F1-score. Uses accuracy, if False.')
  
  args = parser.parse_args()
  DATASET = args.data_set
  PATH_TO_DATASET = args.path_to_data_set
  MINIMUM_BATCH_SIZE = args.minimum_batch_size
  MAXIMUM_BATCH_SIZE = args.maximum_batch_size
  LARGE_LANGUAGE_MODEL = args.large_language_model
  F1_SCORE = args.use_f1_score
  
  if F1_SCORE not in [0, 1]:
    raise ValueError('The F1 score parameter must be either 0 (False) or 1 (True).')

  if MINIMUM_BATCH_SIZE < 1 or MAXIMUM_BATCH_SIZE < 1:
    raise ValueError('The batch sizes must be greater than 0.')
  
  if MINIMUM_BATCH_SIZE >= MAXIMUM_BATCH_SIZE:
    raise ValueError('The minimum batch size must be less than the maximum batch size.')
  
  F1_SCORE = bool(F1_SCORE)
  
  training_df = pd.read_csv(os.path.join(PATH_TO_DATASET, 'train.csv'))
  validation_df = pd.read_csv(os.path.join(PATH_TO_DATASET, 'validation.csv'))  
  testing_df = pd.read_csv(os.path.join(PATH_TO_DATASET, 'test.csv'))

  study_name = f'{DATASET}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}'
  storage = f'sqlite:///optuna_studies/{study_name}.db'
    
  sampler = optuna.samplers.TPESampler(seed = SEED)
  study = optuna.create_study(
    direction = 'maximize',
    sampler = sampler,
    study_name = study_name, 
    storage = storage, 
    load_if_exists = True
  )
  if ('value' in study.trials_dataframe().columns) and (study.trials_dataframe()['value'] >= 0.0).sum() >= TRIALS:
    print('Optimization already completed.', flush = True)
  else:
    study.optimize(objective_function, callbacks = [callback])

  # best_params = study.best_params
  # print('Best hyper-parameters:', best_params, flush = True)
  
  # best_value = study.best_value
  # print('Best validation performance:', best_value, flush = True)
  
  # hyper_parameter_importances = optuna.importance.get_param_importances(study)
  # print(hyper_parameter_importances, flush = True)

  trial_features = [
    'number', 'value', 'params_balanced_loss', 'params_batch_size',
    'params_early_stopping_patience', 'params_epochs', 'params_learning_rate', 'params_plateau_divider',
    'params_plateau_patience', 'params_weight_decay', 'params_beta_0', 'params_beta_1',
    'params_epsilon', 'user_attrs_epoch', 'user_attrs_training_loss', 'user_attrs_validation_loss'
  ]

  top_N_trials = study.trials_dataframe()[trial_features].sort_values(by = ['value', 'user_attrs_validation_loss', 'user_attrs_training_loss'], ascending = [False, True, True]).head(TOP_N)

  for _, trial in top_N_trials.iterrows():
    
    print('\n[TRIAL]', trial['number'], '[VALIDATION PERFORMANCE]', trial['value'], '[TRAINING LOSS]', trial['user_attrs_training_loss'], '[VALIDATION LOSS]', trial['user_attrs_validation_loss'], '\n', flush = True)
    print(trial, flush = True)

    os.makedirs(os.path.join(STORAGE_PATH, f'{DATASET}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}', f'{trial["number"]}'), exist_ok = True)
  
    test_performances = []
    validation_performances = []

    for random_state in range(SEED - NUMBER_OF_TEST_RUNS // 2, SEED + NUMBER_OF_TEST_RUNS // 2):

      if os.path.exists(os.path.join(STORAGE_PATH, f'{DATASET}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}', f'{trial["number"]}', f'{random_state}', 'predictions.csv')) and os.path.isfile(os.path.join(STORAGE_PATH, f'{DATASET}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}', f'{trial["number"]}', f'{random_state}', 'runtimes.csv')):
        
        predictions = pd.read_csv(os.path.join(STORAGE_PATH, f'{DATASET}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}', f'{trial["number"]}', f'{random_state}', 'predictions.csv'))
        
        validation_predictions = predictions[predictions['split'] == 'validation']
        validation_performance = sklearn.metrics.f1_score(validation_predictions['real'], validation_predictions['prediction'], average = 'macro') if F1_SCORE else sklearn.metrics.accuracy_score(validation_predictions['real'], validation_predictions['prediction'])

        test_predictions = predictions[predictions['split'] == 'test']
        test_performance = sklearn.metrics.f1_score(test_predictions['real'], test_predictions['prediction'], average = 'macro') if F1_SCORE else sklearn.metrics.accuracy_score(test_predictions['real'], test_predictions['prediction'])
      else:
        test_performance, validation_performance, _, _ = train_and_predict(
          training_df,
          validation_df,
          testing_df,
          #
          random_state = random_state,
          #
          batch_size = trial['params_batch_size'],
          learning_rate = trial['params_learning_rate'],
          weight_decay = trial['params_weight_decay'],
          beta_0 = trial['params_beta_0'],
          beta_1 = trial['params_beta_1'],
          epsilon = trial['params_epsilon'],
          balanced_loss = trial['params_balanced_loss'],
          epochs = trial['params_epochs'],
          early_stopping_patience = trial['params_early_stopping_patience'],
          plateau_patience = trial['params_plateau_patience'],
          plateau_divider = trial['params_plateau_divider'],
          #
          evaluate_test = True,
          trial_number = trial['number']
        )
      
      if test_performance == -1.0:
        print(random_state, 'Exception...', flush = True)
        continue

      print(random_state, 'Val:', validation_performance, 'Test:', test_performance, flush = True)

      test_performances.append(test_performance)
      validation_performances.append(validation_performance)

    print(
      'Validation performance:',
      round(min(validation_performances) * 100, 2), '&',
      round(statistics.mean(validation_performances) * 100, 2), '±', round(statistics.stdev(validation_performances) * 100, 2), '&',
      round(max(validation_performances) * 100, 2),
      flush = True
    )
    print(
      'Testing performance:',
      round(min(test_performances) * 100, 2), '&',
      round(statistics.mean(test_performances) * 100, 2), '±', round(statistics.stdev(test_performances) * 100, 2), '&',
      round(max(test_performances) * 100, 2),
      flush = True
    )

print(f'\n[{DATASET}] Elapsed time:', (time.time() - st) / 60, 'minutes.', flush = True)
