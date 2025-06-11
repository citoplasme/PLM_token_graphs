import transformers

def load_feature_extractor(model_name : str):
  # -----------------------------------------------------------------------------------
  # -------------------------------------- XLNet --------------------------------------
  # -----------------------------------------------------------------------------------
  if model_name.startswith('xlnet'):
    # MODEL_NAME = 'xlnet/xlnet-base-cased' | 'xlnet/xlnet-large-cased
    # SPECIAL_TOKENS = ['<s>', '</s>', '<unk>', '<sep>', '<pad>', '<cls>', '<mask>', '<eop>', '<eod>']
    chunking = False
    word_piece_special_token = '▁'
    word_piece_special_token_on_word = True
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    llm = transformers.AutoModel.from_pretrained(model_name, output_attentions = True) # output_attentions = True so that output contains attention coefficients
    maximum_chunk_size = 1024 * 10 # None, but set up a value for the experiments
    embedding_output_key = 'last_hidden_state'
    attention_output_key = 'attentions'
  # -----------------------------------------------------------------------------------
  # --------------------------------------- BERT --------------------------------------
  # -----------------------------------------------------------------------------------
  elif model_name.startswith('google-bert'):
    # MODEL_NAME = 'google-bert/bert-base-uncased' | 'google-bert/bert-large-uncased' | 'google-bert/bert-base-cased' | 'google-bert/bert-large-cased'
    # SPECIAL_TOKENS = ['[CLS]', '[SEP]']
    chunking = True
    word_piece_special_token = '##'
    word_piece_special_token_on_word = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    llm = transformers.AutoModel.from_pretrained(model_name, output_attentions = True)
    maximum_chunk_size = 512
    embedding_output_key = 'last_hidden_state'
    attention_output_key = 'attentions'
  # -----------------------------------------------------------------------------------
  # --------------------------------------- BART --------------------------------------
  # -----------------------------------------------------------------------------------
  elif model_name.startswith('facebook/bart'):
    # MODEL_NAME = 'facebook/bart-base' | 'facebook/bart-large'
    # SPECIAL_TOKENS = ['<s>', '</s>', '<unk>', '<pad>', '<mask>']
    chunking = True
    word_piece_special_token = 'Ġ'
    word_piece_special_token_on_word = True
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    llm = transformers.AutoModel.from_pretrained(model_name, attn_implementation = 'eager', output_attentions = True)
    maximum_chunk_size = 1024
    embedding_output_key = 'encoder_last_hidden_state'
    attention_output_key = 'encoder_attentions'
  # -----------------------------------------------------------------------------------
  # -------------------------------- Bio-Clinical BERT --------------------------------
  # -----------------------------------------------------------------------------------
  elif model_name.startswith('emilyalsentzer/Bio_ClinicalBERT'):
    # MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
    # SPECIAL_TOKENS = ['[CLS]', '[SEP]']
    chunking = True
    word_piece_special_token = '##'
    word_piece_special_token_on_word = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    llm = transformers.AutoModel.from_pretrained(model_name, output_attentions = True)
    maximum_chunk_size = 512
    embedding_output_key = 'last_hidden_state'
    attention_output_key = 'attentions'
  # -----------------------------------------------------------------------------------
  # ------------------------------------- RoBERTa -------------------------------------
  # -----------------------------------------------------------------------------------
  elif model_name.startswith('FacebookAI/roberta'):
    # MODEL_NAME = 'FacebookAI/roberta-base' | 'FacebookAI/roberta-large'
    # SPECIAL_TOKENS = ['<s>', '</s>', '<unk>', '<pad>', '<mask>']
    chunking = True
    word_piece_special_token = 'Ġ'
    word_piece_special_token_on_word = True
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    llm = transformers.AutoModel.from_pretrained(model_name, output_attentions = True)
    maximum_chunk_size = 512
    embedding_output_key = 'last_hidden_state'
    attention_output_key = 'attentions'
  else:
    raise ValueError('Model not supported.')
  return chunking, word_piece_special_token, word_piece_special_token_on_word, tokenizer, llm, maximum_chunk_size, embedding_output_key, attention_output_key
