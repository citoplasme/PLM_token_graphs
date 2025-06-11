import pandas as pd
import spacy
import textstat
from collections import Counter

import time

# https://pypi.org/project/textstat/
def flesch_reading_ease(text):
  flesch_reading_score = textstat.flesch_reading_ease(text)
  if flesch_reading_score >= 90:
    return 'Very Easy'
  elif flesch_reading_score >= 80:
    return 'Easy'
  elif flesch_reading_score >= 70:
    return 'Fairly Easy'
  elif flesch_reading_score >= 60:
    return 'Standard'
  elif flesch_reading_score >= 50:
    return 'Fairly Difficult'
  elif flesch_reading_score >= 30:
    return 'Difficult'
  else:
    return 'Very Confusing'


st = time.time()

# Load spaCy model
#nlp = spacy.load('en_core_web_lg')
nlp = spacy.load('en_core_web_sm')

df = None
for split in ['train', 'validation', 'test']:
  split_df = pd.read_csv(f'../data/with_validation_splits/R8/{split}.csv').assign(split = split).reset_index()
  if df is None:
    df = split_df
  else:
    df = pd.concat([df, split_df], axis = 0)

d = {
  'split' : list(),
  'index' : list(),
  #'lexical_density' : list(),
  'average_word_length' : list(),
  'average_sentence_length' : list(),
  'Flesch_Reading_ease_score' : list(),
  'average_syllables_per_word' : list(),
  'percentage_monosyllabic_words' : list(),
  'percentage_polysyllabic_words' : list(),
  'text_length' : list(),
  'type-token_ratio' : list()
}

pos_l = list()

for _, row in df.iterrows():

  text = row['text']
  doc = nlp(text)
  words = [token.text for token in doc if token.is_alpha] # CONFIRM THIS
  unique_words = set(words)
  sentences = list(doc.sents)

  d['split'].append(row['split'])
  d['index'].append(row['index'])

  # Text length
  text_length = len(words)
  d['text_length'].append(text_length)
  # Type-token ratio (TTR)
  ttr = len(unique_words) / text_length
  d['type-token_ratio'].append(ttr)
  # Lexical density
  #content_words = [token for token in doc if token.is_alpha and not token.is_stop]
  #lexical_density = len(content_words) / text_length
  #d['lexical_density'].append(lexical_density)
  # Average word length
  average_word_length = sum(len(word) for word in words) / text_length
  d['average_word_length'].append(average_word_length)
  # Average word count per sentence
  average_sentence_length = text_length / len(sentences)
  d['average_sentence_length'].append(average_sentence_length)
  # Readability score
  flesch_reading_ease_score = flesch_reading_ease(text)
  d['Flesch_Reading_ease_score'].append(flesch_reading_ease_score)
  # POS tagging distribution
  pos_counts = Counter([token.pos_ for token in doc])
  pos_distribution = {pos: count / text_length for pos, count in pos_counts.items()}
  pos_l.append(pos_distribution)
  # Average syllable count per word
  syllable_count = textstat.syllable_count(text) / text_length
  d['average_syllables_per_word'].append(syllable_count)
  # Percentage of monosyllabic words
  monosyllabic_word_count = textstat.monosyllabcount(text) / text_length
  d['percentage_monosyllabic_words'].append(monosyllabic_word_count)
  # Percentage of polysyllabic (3 or more syllables) words
  polysyllabic_word_count = textstat.polysyllabcount(text) / text_length
  d['percentage_polysyllabic_words'].append(polysyllabic_word_count)

df = pd.DataFrame(d)
pos_df = pd.DataFrame(pos_l)[['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'PRON']].fillna(0)#.add_prefix('POS_tag_')
all_df = pd.concat([df, pos_df], axis = 1)

et = time.time()

pd.set_option('display.max_columns', None)
print(all_df.head(5), flush = True)
print(f'Elapsed time: {et - st}', flush = True)


