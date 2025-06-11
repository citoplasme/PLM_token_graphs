import os
import torch
import numpy as np
import pandas as pd
import time
import argparse
import optuna
import torch_geometric
import sklearn.model_selection
import sklearn.metrics
import sklearn.utils
import random
import gc
import re

import utils
import construct_graphs
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

CHECKPOINT_PATH = f'/checkpoint/pimentel/{os.environ["SLURM_JOB_ID"]}/'

# ===========================================================================
# ============================= Auxiliary methods ===========================
# ===========================================================================

def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

def calculate_node_importance(edge_index, edge_mask, node_count):
  node_importance = torch.zeros(node_count, device = edge_mask.device)
  for i, (src, dst) in enumerate(edge_index.t()):
    importance = edge_mask[i]
    node_importance[src] += importance
    node_importance[dst] += importance
  return node_importance

def get_node_and_edge_importance(data, edge_mask, node_mask):
  edge_index = data.edge_index.cpu().numpy()
  edge_mask = edge_mask.cpu().detach().numpy()
  node_mask = node_mask.cpu().detach().numpy()
  node_tokens = data.node_tokens[0]

  node_df = pd.DataFrame({
    'ids' : [i for i in range(len(node_tokens))],
    'tokens': node_tokens,
    'aggregated_weights': node_mask
  }).sort_values('aggregated_weights', ascending = False)
  
  src, dst = edge_index
  edge_df = pd.DataFrame({
    'from_ids': [i for i in src],
    'from_token': [node_tokens[i] for i in src],
    'to_ids': [i for i in dst],
    'to_token': [node_tokens[i] for i in dst],
    'weights': edge_mask
  }).sort_values('weights', ascending = False)

  return node_df, edge_df

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
    chunk_size,
    left_stride,
    right_stride,
    attention_pooling_operation,
    embedding_pooling_operation,
    threshold,
    #
    batch_size,
    attention_heads,
    hidden_dimension,
    number_of_hidden_layers,
    dropout_rate,
    global_pooling,
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

    training_graphs_pyg = list()
    validation_graphs_pyg = list()
    testing_graphs_pyg = list()
    
    for i, row in training_df.iterrows():
      training_graphs_pyg.append(
        construct_graphs.construct_PyG_graph_from_LLM(
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
          attention_pooling = attention_pooling_operation,
          embedding_pooling = embedding_pooling_operation,
          threshold = threshold,
          aggregation_level = AGGREGATION_LEVEL,
          #
          llm = LLM,
          tokenizer = TOKENIZER,
          attention_output_key = ATTENTION_OUTPUT_KEY,
          embedding_output_key = EMBEDDING_OUTPUT_KEY,
          maximum_chunk_size = MAXIMUM_SEQUENCE_LENGTH,
          layers = LAYERS,
          heads = HEADS,
          #
          device = DEVICE
        )
      )

    for i, row in validation_df.iterrows():
      validation_graphs_pyg.append(
        construct_graphs.construct_PyG_graph_from_LLM(
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
          attention_pooling = attention_pooling_operation,
          embedding_pooling = embedding_pooling_operation,
          threshold = threshold,
          aggregation_level = AGGREGATION_LEVEL,
          #
          llm = LLM,
          tokenizer = TOKENIZER,
          attention_output_key = ATTENTION_OUTPUT_KEY,
          embedding_output_key = EMBEDDING_OUTPUT_KEY,
          maximum_chunk_size = MAXIMUM_SEQUENCE_LENGTH,
          layers = LAYERS,
          heads = HEADS,
          #
          device = DEVICE
        )
      )

    for i, row in testing_df.iterrows():
      testing_graphs_pyg.append(
        construct_graphs.construct_PyG_graph_from_LLM(
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
          attention_pooling = attention_pooling_operation,
          embedding_pooling = embedding_pooling_operation,
          threshold = threshold,
          aggregation_level = AGGREGATION_LEVEL,
          #
          llm = LLM,
          tokenizer = TOKENIZER,
          attention_output_key = ATTENTION_OUTPUT_KEY,
          embedding_output_key = EMBEDDING_OUTPUT_KEY,
          maximum_chunk_size = MAXIMUM_SEQUENCE_LENGTH,
          layers = LAYERS,
          heads = HEADS,
          #
          device = DEVICE
        )
      )
    
    # ===========================================================================
    # ============================== Batchification =============================
    # ===========================================================================

    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())

    training_batches = torch_geometric.loader.DataLoader(training_graphs_pyg, batch_size = batch_size, num_workers = 0, worker_init_fn = seed_worker, generator = generator, shuffle = True)
    validation_batches = torch_geometric.loader.DataLoader(validation_graphs_pyg, batch_size = batch_size, num_workers = 0, worker_init_fn = seed_worker, generator = generator, shuffle = True)
    testing_batches = torch_geometric.loader.DataLoader(testing_graphs_pyg, batch_size = batch_size, num_workers = 0, worker_init_fn = seed_worker, generator = generator, shuffle = True)

    # ===========================================================================
    # ================================= Training ================================
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
        optimizer.zero_grad()

        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        loss = criterion(outputs, batch.y)
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
          outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
          predictions = outputs.argmax(dim = 1)

          loss = criterion(outputs, batch.y)
          total_validation_loss += loss.item()

          validation_predictions.extend(predictions.cpu().numpy())
          validation_labels.extend(batch.y.cpu().numpy())
          validation_indices.extend(batch.identifier.cpu().numpy()) # validation_indices.extend(batch.index.cpu().numpy())

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

    print('Completed training the base model.', flush = True)
    
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
          
          outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
          predictions = outputs.argmax(dim = 1)

          evaluation_runtime = evaluation_runtime + (time.time() - evaluation_start_time)

          test_predictions.extend(predictions.cpu().numpy())
          test_labels.extend(batch.y.cpu().numpy())
          #test_probability.extend(outputs.cpu().numpy())
          test_indices.extend(batch.identifier.cpu().numpy()) # test_indices.extend(batch.index.cpu().numpy())
      
      # Average evaluation time per instance
      average_evaluation_runtime = evaluation_runtime / len(testing_graphs_pyg)

      # PGExplainer training
      PGEXPLAINER_EPOCHS = 30
      explainer = torch_geometric.explain.Explainer(
        model = model,
        algorithm = torch_geometric.explain.PGExplainer(epochs = PGEXPLAINER_EPOCHS, lr = 0.003), 
        explanation_type = 'phenomenon',
        model_config = torch_geometric.explain.config.ModelConfig(
          mode = 'multiclass_classification',
          task_level = 'graph',
          return_type = 'log_probs'
        ),
        edge_mask_type = 'object',
      )
      explainer.algorithm.to(DEVICE)

      for epoch in range(PGEXPLAINER_EPOCHS):
        for batch in training_batches:
          batch = batch.to(DEVICE)
          explainer.algorithm.train(
            epoch,
            model,
            x = batch.x,
            edge_index = batch.edge_index,
            edge_attr = batch.edge_attr,
            batch = batch.batch,
            target = batch.y
          )
      print('Completed training the PGExplainer model.', flush = True)

      # PGExplainer inference
      explainability_documents = [x for x in os.listdir(os.path.join('.', 'data', f'{LARGE_LANGUAGE_MODEL.replace("/", "-")}', DATASET)) if x.endswith(f'{"Surrogate" if SURROGATE else "Grouped"}.csv')]
      explainability_index = int(re.sub(r'-.*\.csv', '', explainability_documents[0]))
      explainability_testing_batches = torch_geometric.loader.DataLoader([testing_graphs_pyg[explainability_index]], batch_size = 1, num_workers = 0, worker_init_fn = seed_worker, generator = generator, shuffle = True)

      explainability_graph = next(iter(explainability_testing_batches)).to(DEVICE)
      explanation = explainer(
        x = explainability_graph.x,
        edge_index = explainability_graph.edge_index,
        edge_attr = explainability_graph.edge_attr,
        batch = explainability_graph.batch,
        index = 0,
        target = explainability_graph.y
      )

      node_mask = calculate_node_importance(explainability_graph.edge_index, explanation.edge_mask, explainability_graph.num_nodes)
      node_df, edge_df = get_node_and_edge_importance(explainability_graph, explanation.edge_mask, node_mask)

      node_df.to_csv(os.path.join('.', 'data', f'{LARGE_LANGUAGE_MODEL.replace("/", "-")}', DATASET, f'{explainability_index}-{"Surrogate" if SURROGATE else "Grouped"}-node_importance.csv'), index = False)
      edge_df.to_csv(os.path.join('.', 'data', f'{LARGE_LANGUAGE_MODEL.replace("/", "-")}', DATASET, f'{explainability_index}-{"Surrogate" if SURROGATE else "Grouped"}-edge_importance.csv'), index = False)

      # Remove model from GPU
      del model

    return 0
  except Exception as e:
    print(e, flush = True)
    return -1.0

# ===========================================================================
# ========================== Information extraction =========================
# ===========================================================================

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_set', required = True, type = str, help = 'The name of the data set.')
  parser.add_argument('--path_to_data_set', required = True, type = str, help = 'The path to the data set\'s training, validation, and testing files.')
  parser.add_argument('--chunking', required = True, type = int, help = 'Whether or not to chunk the input text depending on the maximum sequence length accepted by the model and the properties of the text.')
  parser.add_argument('--minimum_threshold', required = True, type = float, help = 'The minimum value for threshold applied to the attention coefficients during hyper-parameter tuning.')
  parser.add_argument('--maximum_threshold', required = True, type = float, help = 'The maximum value for threshold applied to the attention coefficients during hyper-parameter tuning.')
  parser.add_argument('--minimum_batch_size', required = True, type = float, help = 'The minimum value for batch size used during hyper-parameter tuning.')
  parser.add_argument('--maximum_batch_size', required = True, type = float, help = 'The maximum value for batch size used during hyper-parameter tuning.')
  parser.add_argument('--large_language_model', required = True, type = str, help = 'Which large language model to use to construct word graphs.')
  parser.add_argument('--use_f1_score', required = True, type = int, help = 'Whether or not to evaluate the models using macro F1-score. Uses accuracy, if False.')
  parser.add_argument('--graph_neural_network', required = True, type = str, help = 'Which graph neural network to use during the learning step.')
  parser.add_argument('--surrogate', required = True, type = int, help = 'Whether or not to follow a surrogate approach to construct the graphs, i.e., grouping the nodes based on their indices instead of their identifiers.')
  parser.add_argument('--aggregation_level', required = True, type = int, help = 'Which level of aggregation to apply to the attention coefficients: 0 (no aggregation), 1 (layer-wise aggregation), or 2 (global aggregation).')

  args = parser.parse_args()
  DATASET = args.data_set
  PATH_TO_DATASET = args.path_to_data_set
  CHUNKING = args.chunking
  MINIMUM_THRESHOLD = args.minimum_threshold
  MAXIMUM_THRESHOLD = args.maximum_threshold
  MINIMUM_BATCH_SIZE = args.minimum_batch_size
  MAXIMUM_BATCH_SIZE = args.maximum_batch_size
  LARGE_LANGUAGE_MODEL = args.large_language_model
  F1_SCORE = args.use_f1_score
  GNN = args.graph_neural_network
  SURROGATE = args.surrogate
  AGGREGATION_LEVEL = args.aggregation_level

  if CHUNKING not in [0, 1]:
    raise ValueError('The chunking parameter must be either 0 (False) or 1 (True).')

  if SURROGATE not in [0, 1]:
    raise ValueError('The surrogate parameter must be either 0 (False) or 1 (True).')
  
  if F1_SCORE not in [0, 1]:
    raise ValueError('The F1 score parameter must be either 0 (False) or 1 (True).')

  if GNN not in ['GAT', 'GATv2']:
    raise ValueError('The selected GNN is not supported. Please choose either GAT or GATv2.')

  if AGGREGATION_LEVEL not in [0, 1, 2]:
    raise ValueError('The aggregation level must be 0 (no aggregation), 1 (layer-wise aggregation), or 2 (global aggregation).')
  
  if MINIMUM_THRESHOLD < 0 or MINIMUM_THRESHOLD > 1 or MAXIMUM_THRESHOLD < 0 or MAXIMUM_THRESHOLD > 1:
    raise ValueError('The thresholds must be between 0 and 1.')

  if MINIMUM_THRESHOLD >= MAXIMUM_THRESHOLD:
    raise ValueError('The minimum threshold must be less than the maximum threshold.')

  if MINIMUM_BATCH_SIZE < 1 or MAXIMUM_BATCH_SIZE < 1:
    raise ValueError('The batch sizes must be greater than 0.')
  
  if MINIMUM_BATCH_SIZE >= MAXIMUM_BATCH_SIZE:
    raise ValueError('The minimum batch size must be less than the maximum batch size.')
  
  CHUNKING = bool(CHUNKING)
  SURROGATE = bool(SURROGATE)
  F1_SCORE = bool(F1_SCORE)

  _, _, _, TOKENIZER, LLM, MAXIMUM_SEQUENCE_LENGTH, EMBEDDING_OUTPUT_KEY, ATTENTION_OUTPUT_KEY = utils.load_feature_extractor(LARGE_LANGUAGE_MODEL)
  LLM.to(DEVICE)

  training_df = pd.read_csv(os.path.join(PATH_TO_DATASET, 'train.csv'))
  validation_df = pd.read_csv(os.path.join(PATH_TO_DATASET, 'validation.csv'))  
  testing_df = pd.read_csv(os.path.join(PATH_TO_DATASET, 'test.csv'))

  EMBEDDING_DIMENSION = LLM.config.hidden_size
  LAYERS = LLM.config.num_hidden_layers
  HEADS = LLM.config.num_attention_heads

  AGGREGATION_LEVELS = ['No_Aggregation', 'Layer-wise_Aggregation', 'Global_Aggregation']
  AGGREGATION_LEVELS_FEATURE_COUNT = [LAYERS * HEADS, LAYERS, 1]
  
  study_name = f'{DATASET}-{GNN}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}-{"Surrogate" if SURROGATE else "Grouped"}-{AGGREGATION_LEVELS[AGGREGATION_LEVEL]}'
  storage = f'sqlite:///../pipelines/optuna_studies/{study_name}.db'
    
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

  if CHUNKING:
    trial_features += ['params_left_stride', 'params_right_stride'] #['params_chunk_size', 'params_left_stride', 'params_right_stride']

  top_N_trials = study.trials_dataframe()[trial_features].sort_values(by = ['value', 'user_attrs_validation_loss', 'user_attrs_training_loss'], ascending = [False, True, True]).head(TOP_N)

  for _, trial in top_N_trials.iterrows():
    
    random_states = [x for x in os.listdir(os.path.join(STORAGE_PATH, f'{DATASET}-{GNN}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}', 'Surrogate' if SURROGATE else 'Grouped', AGGREGATION_LEVELS[AGGREGATION_LEVEL], f'{trial["number"]}')) if os.path.isdir(os.path.join(STORAGE_PATH, f'{DATASET}-{GNN}-{LARGE_LANGUAGE_MODEL.replace("/", "-")}', 'Surrogate' if SURROGATE else 'Grouped', AGGREGATION_LEVELS[AGGREGATION_LEVEL], f'{trial["number"]}', f'{x}'))]
    if len(random_states) < NUMBER_OF_TEST_RUNS:
      continue
    
    print('\n[TRIAL]', trial['number'], '[VALIDATION PERFORMANCE]', trial['value'], '[TRAINING LOSS]', trial['user_attrs_training_loss'], '[VALIDATION LOSS]', trial['user_attrs_validation_loss'], '\n', flush = True)
    print(trial, flush = True)

    train_and_predict(
      training_df,
      validation_df,
      testing_df,
      #
      random_state = SEED,
      #
      chunk_size = MAXIMUM_SEQUENCE_LENGTH, # trial['params_chunk_size'] if CHUNKING else MAXIMUM_SEQUENCE_LENGTH,
      left_stride = trial['params_left_stride'] if CHUNKING else 0,
      right_stride = trial['params_right_stride'] if CHUNKING else 0,
      attention_pooling_operation = trial['params_attention_pooling_operation'],
      embedding_pooling_operation = trial['params_embedding_pooling_operation'],
      threshold = trial['params_threshold'],
      #
      batch_size = trial['params_batch_size'],
      attention_heads = trial['params_attention_heads'],
      hidden_dimension = trial['params_hidden_dimension'],
      number_of_hidden_layers = trial['params_number_of_hidden_layers'],
      dropout_rate = trial['params_dropout_rate'],
      global_pooling = trial['params_global_pooling'],
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
    break

print(f'\n[{DATASET}] Elapsed time:', (time.time() - st) / 60, 'minutes.', flush = True)
