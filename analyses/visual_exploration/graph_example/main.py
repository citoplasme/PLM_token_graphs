import torch
import pandas as pd
import construct_graphs
import utils

SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LARGE_LANGUAGE_MODEL = 'google-bert/bert-base-uncased'

TEXT = 'The quick brown fox jumps over the lazy dog'

_, _, _, TOKENIZER, LLM, MAXIMUM_SEQUENCE_LENGTH, EMBEDDING_OUTPUT_KEY, ATTENTION_OUTPUT_KEY = utils.load_feature_extractor(LARGE_LANGUAGE_MODEL)
LLM.to(DEVICE)

EMBEDDING_DIMENSION = LLM.config.hidden_size
LAYERS = LLM.config.num_hidden_layers
HEADS = LLM.config.num_attention_heads

for surrogate in [False, True]:
  for thresold in [0.0, 0.6]: # 0.0 = no filtering, 0.6 = roughly the best value in SST-2 (short documents) for both Surrogate and Grouped
    # Modified version of the method that does not create a PyG graph so that the tokens can be recovered from indices.
    from_indices, from_tokens, to_indices, to_tokens, edge_attrs = construct_graphs.construct_PyG_graph_from_LLM(
      text = TEXT,
      label = 0, # Not used
      split = 'train', # Not used
      index = 0, # Not used
      #
      chunk_size = MAXIMUM_SEQUENCE_LENGTH,
      left_stride = 0, # Not used
      right_stride = 0, # Not used
      #
      surrogate = surrogate,
      attention_pooling = 'mean',
      embedding_pooling = 'mean',
      threshold = thresold,
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
    pd.DataFrame({'from_id' : from_indices, 'from' : from_tokens, 'to_id' : to_indices, 'to' : to_tokens, 'weight' : edge_attrs}).to_csv(f'./{"Surrogate" if surrogate else "Grouped"}-{thresold * 100}%.csv', index = False)