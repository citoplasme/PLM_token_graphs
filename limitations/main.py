import os
import re
import time
import torch
import numpy as np
import pandas as pd
import random
import gc

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/h/pimentel/DynamicCOO/src/pipelines/')
import utils
import construct_graphs

# ===========================================================================
# ======================== Cache clearing and seeding =======================
# ===========================================================================

SEED = 42

gc.collect()
torch.cuda.empty_cache()

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===========================================================================
# ============================ Global parameters ============================
# ===========================================================================

N = 300
RUNS = 3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if os.path.exists('./random_sample.csv'):
  df = pd.read_csv('./random_sample.csv')
else:
  df = pd.concat([
    pd.read_csv('../../data/with_validation_splits/SST-2/validation.csv').reset_index(names = 'index').assign(dataset = 'SST-2', split = 'validation')[['dataset', 'index', 'split', 'text']],
    pd.read_csv('../../data/with_validation_splits/SST-2/test.csv').reset_index(names = 'index').assign(dataset = 'SST-2', split = 'test')[['dataset', 'index', 'split', 'text']],
    pd.read_csv('../../data/with_validation_splits/Ohsumed/validation.csv').reset_index(names = 'index').assign(dataset = 'Ohsumed', split = 'validation')[['dataset', 'index', 'split', 'text']],
    pd.read_csv('../../data/with_validation_splits/Ohsumed/test.csv').reset_index(names = 'index').assign(dataset = 'Ohsumed', split = 'test')[['dataset', 'index', 'split', 'text']],
    pd.read_csv('../../data/with_validation_splits/R8/validation.csv').reset_index(names = 'index').assign(dataset = 'R8', split = 'validation')[['dataset', 'index', 'split', 'text']],
    pd.read_csv('../../data/with_validation_splits/R8/test.csv').reset_index(names = 'index').assign(dataset = 'R8', split = 'test')[['dataset', 'index', 'split', 'text']],
    pd.read_csv('../../data/with_validation_splits/IMDb-top_1000/validation.csv').reset_index(names = 'index').assign(dataset = 'IMDb-top_1000', split = 'validation')[['dataset', 'index', 'split', 'text']],
    pd.read_csv('../../data/with_validation_splits/IMDb-top_1000/test.csv').reset_index(names = 'index').assign(dataset = 'IMDb-top_1000', split = 'test')[['dataset', 'index', 'split', 'text']],
  ]).sample(N, random_state = SEED).reset_index(drop = True)
  df.to_csv('./random_sample.csv', index = False)

# ===========================================================================
# =========================== Auxiliary functions ===========================
# ===========================================================================

# Used MPAD, TextING, and Text-level GNN
def clean_str(string):
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
  string = re.sub(r"\'s", " \'s", string) 
  string = re.sub(r"\'ve", " \'ve", string) 
  string = re.sub(r"n\'t", " n\'t", string) 
  string = re.sub(r"\'re", " \'re", string) 
  string = re.sub(r"\'d", " \'d", string) 
  string = re.sub(r"\'ll", " \'ll", string) 
  string = re.sub(r",", " , ", string) 
  string = re.sub(r"!", " ! ", string) 
  string = re.sub(r"\(", " \( ", string) 
  string = re.sub(r"\)", " \) ", string) 
  string = re.sub(r"\?", " \? ", string) 
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower().split()

# From https://github.com/CRIPAC-DIG/TextING/tree/master
def texting_graphs(text, window_size):
  text = clean_str(text)
  text_length = len(text)

  vocabulary = list(set(text))
  nodes = len(vocabulary)

  word_id_map = {}
  for j in range(nodes):
    word_id_map[vocabulary[j]] = j

  windows = []
  if text_length <= window_size:
    windows.append(text)
  else:
    for j in range(text_length - window_size + 1):
      window = text[j: j + window_size]
      windows.append(window)

  edges = {}
  for window in windows:
    for p in range(1, len(window)):
      for q in range(0, p):
        word_p = window[p]
        word_p_id = word_id_map[word_p]
        word_q = window[q]
        word_q_id = word_id_map[word_q]
        if word_p_id == word_q_id:
          continue
        word_pair_key = (word_p_id, word_q_id)
        # word co-occurrences as weights
        if word_pair_key in edges:
          edges[word_pair_key] += 1.
        else:
          edges[word_pair_key] = 1.
        # bi-direction
        word_pair_key = (word_p_id, word_q_id)
        if word_pair_key in edges:
          edges[word_pair_key] += 1.
        else:
          edges[word_pair_key] = 1.
  
  return nodes, edges

# From https://github.com/giannisnik/mpad
def mpad_graphs(text, window_size, use_master_node = True, directed = True):
  text = clean_str(text)
  edges = dict()
  idx = dict()
  l_terms = list()

  for i in range(len(text)):
    if text[i] not in idx:
        l_terms.append(text[i])
        idx[text[i]] = len(idx)
  
  if use_master_node:
    idx['master_node'] = len(idx)
  
  for i in range(len(text)):
    for j in range(i + 1, i + window_size):
      if j < len(text):
        if (text[i], text[j]) in edges:
          edges[(text[i], text[j])] += 1.0/(j-i)
          if not directed:
            edges[(text[j], text[i])] += 1.0/(j-i)
        else:
          edges[(text[i], text[j])] = 1.0/(j-i)
          if not directed:
            edges[(text[j], text[i])] = 1.0/(j-i)
      if use_master_node:
          edges[(text[i], 'master_node')] = 1.0
          edges[('master_node', text[i])] = 1.0
  return list(idx.keys()), edges

# ===========================================================================
# =========================== Graph construction ============================
# ===========================================================================

if os.path.exists('./sliding_windows.csv'):
  print('Sliding window analysis already complete.', flush = True)
else:
  print('Sliding window analysis', flush = True)

  sliding_window_properties = list()
  for i, row in df.iterrows():
    print(f'{i + 1} / {N}', flush = True)
    for window_size in range(2, 6):
      for j in range(RUNS):
        start_time = time.time()
        nodes, edges = mpad_graphs(row['text'], window_size = window_size)
        time_to_construct = time.time() - start_time
        sliding_window_properties.append((row['dataset'], row['index'], row['split'], window_size, 'MPAD', time_to_construct, len(nodes), len(edges), len(edges) / (len(nodes) * len(nodes))))

        start_time = time.time()
        nodes, edges = texting_graphs(row['text'], window_size = window_size)
        time_to_construct = time.time() - start_time
        sliding_window_properties.append((row['dataset'], row['index'], row['split'], window_size, 'TextING', time_to_construct, nodes, len(edges), len(edges) / (nodes * nodes)))

  sliding_window_times_df = pd.DataFrame(sliding_window_properties, columns = ['dataset', 'index', 'split', 'window_size', 'approach', 'time', 'nodes', 'edges', 'density'])
  sliding_window_times_df.to_csv('./sliding_windows.csv', index = False)






for llm in ['facebook/bart-large', 'facebook/bart-base', 'google-bert/bert-base-uncased', 'FacebookAI/roberta-large']:  
  if os.path.exists(f'./{llm.replace("/", "-")}.csv'):
    print(f'LLM-based method analysis already complete ({llm}).', flush = True)
    continue
  else:
    print(f'LLM-based method analysis ({llm})', flush = True)

    llm_based_properties = list()
    _, _, _, TOKENIZER, LLM, MAXIMUM_SEQUENCE_LENGTH, EMBEDDING_OUTPUT_KEY, ATTENTION_OUTPUT_KEY = utils.load_feature_extractor(llm)
    LLM.to(DEVICE)
    
    EMBEDDING_DIMENSION = LLM.config.hidden_size
    LAYERS = LLM.config.num_hidden_layers
    HEADS = LLM.config.num_attention_heads
  
    for i, row in df.iterrows():
      print(f'{i + 1} / {N}', flush = True)
      for surrogate in [False, True]:
        for left_stride in [64, 128]:
          for right_stride in [64, 128]:
            for threshold in [0.7, 0.85, 0.95]:
              for j in range(RUNS):

                gc.collect()
                torch.cuda.empty_cache()

                start_time = time.time()
                graph = construct_graphs.construct_PyG_graph_from_LLM(
                  text = row['text'],
                  label = 0, # Not used for this analysis
                  split = row['split'],
                  index = i,
                  #
                  chunk_size = MAXIMUM_SEQUENCE_LENGTH,
                  left_stride = left_stride,
                  right_stride = right_stride,
                  #
                  surrogate = surrogate,
                  attention_pooling = 'mean',
                  embedding_pooling = 'mean',
                  threshold = threshold,
                  aggregation_level = 0,
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
                time_to_construct = time.time() - start_time
                llm_based_properties.append((
                  row['dataset'],
                  row['index'],
                  row['split'],
                  llm,
                  'Surrogate' if surrogate else 'Grouped',
                  left_stride,
                  right_stride,
                  threshold,
                  time_to_construct,
                  graph.x.size(0),
                  graph.edge_index.size(1),
                  graph.edge_index.size(1) / (graph.x.size(0) * graph.x.size(0))
                ))

    llm_based_times_df = pd.DataFrame(llm_based_properties, columns = ['dataset', 'index', 'split', 'LLM', 'approach', 'left_stride', 'right_stride', 'threshold', 'time', 'nodes', 'edges', 'density'])
    llm_based_times_df.to_csv(f'./{llm.replace("/", "-")}.csv', index = False)
