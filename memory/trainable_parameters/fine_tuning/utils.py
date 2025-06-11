import transformers

def load_feature_extractor(model_name : str, num_labels : int):
  # -----------------------------------------------------------------------------------
  # -------------------------------------- XLNet --------------------------------------
  # -----------------------------------------------------------------------------------
  if model_name.startswith('xlnet'):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    llm = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_labels)
    maximum_chunk_size = 1024 * 10 # None, but set up a value for the experiments
  # -----------------------------------------------------------------------------------
  # --------------------------------------- BERT --------------------------------------
  # -----------------------------------------------------------------------------------
  elif model_name.startswith('google-bert'):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    llm = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_labels)
    maximum_chunk_size = 512
  # -----------------------------------------------------------------------------------
  # --------------------------------------- BART --------------------------------------
  # -----------------------------------------------------------------------------------
  elif model_name.startswith('facebook/bart'):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    llm = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_labels, attn_implementation = 'eager')
    maximum_chunk_size = 1024
  # -----------------------------------------------------------------------------------
  # -------------------------------- Bio-Clinical BERT --------------------------------
  # -----------------------------------------------------------------------------------
  elif model_name.startswith('emilyalsentzer/Bio_ClinicalBERT'):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    llm = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_labels)
    maximum_chunk_size = 512
  # -----------------------------------------------------------------------------------
  # ------------------------------------- RoBERTa -------------------------------------
  # -----------------------------------------------------------------------------------
  elif model_name.startswith('FacebookAI/roberta'):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    llm = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_labels)
    maximum_chunk_size = 512
  else:
    raise ValueError('Model not supported.')
  return tokenizer, llm, maximum_chunk_size
