import torch
import pandas as pd
import utils

SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LARGE_LANGUAGE_MODEL = 'google-bert/bert-base-uncased'

TEXT = pd.read_csv('../../../data/with_validation_splits/Ohsumed/test.csv').sample(n = 1, random_state = SEED)['text'].values[0]
print(TEXT, flush = True)

# Adapted from https://github.com/jessevig/bertviz/blob/master/bertviz/head_view.py
def format_attention(attention_coefficients):
  squeezed = []
  for layer_attention in attention_coefficients:
    if len(layer_attention.shape) != 4: # 1, #heads, sequence length, sequence length
      raise ValueError('The attention tensor does not have the correct number of dimensions: 1, number of attention heads, sequence length, sequence length.')
    layer_attention = layer_attention.squeeze(0)
    squeezed.append(layer_attention)
  return torch.stack(squeezed) # #layers, #heads, sequence length, sequence length

_, _, _, TOKENIZER, LLM, MAXIMUM_SEQUENCE_LENGTH, embedding_output_key, attention_output_key = utils.load_feature_extractor(LARGE_LANGUAGE_MODEL)
LLM.to(DEVICE)

# EMBEDDING_DIMENSION = LLM.config.hidden_size
LAYERS = LLM.config.num_hidden_layers
HEADS = LLM.config.num_attention_heads

input_identifiers = TOKENIZER.encode(TEXT, return_tensors = 'pt').to(DEVICE)
special_token_masking = torch.tensor(TOKENIZER.get_special_tokens_mask(input_identifiers[0], already_has_special_tokens = True), dtype = torch.bool).to(DEVICE)
tokens = TOKENIZER.convert_ids_to_tokens(input_identifiers[0])

with torch.no_grad():
  outputs = LLM(input_identifiers)
  attention = format_attention(outputs[attention_output_key])

attention_T = torch.movedim(attention, source = (2, 3), destination = (0, 1)).reshape((input_identifiers.size(1), input_identifiers.size(1), LAYERS * HEADS))

# Pooling attention to a single value for visual purposes
attention_aggregated = torch.mean(attention_T, dim = 2, keepdim = True).reshape(-1,).cpu().numpy()

l = list()
i = 0
for x in input_identifiers[0].cpu().numpy():
  for y in input_identifiers[0].cpu().numpy():
    l.append((x, y, attention_aggregated[i]))
    i = i + 1

pd.DataFrame({'input_identifier' : input_identifiers[0].cpu().numpy(), 'special_token' : special_token_masking.cpu().numpy(), 'token' : tokens}).to_csv('./tokens.csv', index = False)
pd.DataFrame(l, columns = ['from', 'to', 'weight']).to_csv('./attentions.csv', index = False)