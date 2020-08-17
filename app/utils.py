from bs4 import BeautifulSoup
import contractions
import datetime
import gensim
import json
import matplotlib.pyplot as plt

import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np

import pandas as pd
import re, string, unicodedata

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

import time
import torch



# =================== #
# corpus modification #
# =================== #

def denoise_text(text):
	""" Remove html tags and square brackets."""
	if any(text.startswith(prefix) for prefix in ("http:", "https:")):
		text = ""
	else:
		text = BeautifulSoup(text, "html.parser").get_text()
		text = re.sub('\[[^]]*\]', '', text)
	return text

def remove_special_characters(text, remove_digits=True):
	""" Removes special characters. """
	pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
	text = re.sub(pattern, '', text)
	return text


def plot_word_embeddings(model, wordlist, figsize=(12, 8)):
	""" Plots word embeddings. """
	words = []
	for term in wordlist:
		words += [w[0] for w in model.wv.most_similar([term], topn=5)]                                   
	words += wordlist    

	vectors = model.wv[words]

	tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=7)
	T = tsne.fit_transform(vectors)

	plt.figure(figsize=figsize)
	plt.scatter(T[:, 0], T[:, 1])
	for label, x, y in zip(words, T[:, 0], T[:, 1]):
		plt.annotate(label, xy=(x+2, y+2), xytext=(0, 0), textcoords='offset points')

def preprocess_text(text, remove_stopwords=False):
	""" Preprocessing text."""
	text = text.lower()
	text = re.sub('[^a-zA-Z]', ' ', text )
	text = re.sub(r'\s+', ' ', text)
	
	if remove_stopwords:
		text = word_tokenize(text)
		textlist = [w for w in text if w not in stopwords.words('english')]
		text = " ".join(textlist)
	
	return text

def fix_contractions(text):
	""" Dissolve language contractions, e.g.:
		I'm a student -> I am a student
	"""
	return contractions.fix(text)

def clean_text(text):
	""" Cleans text by applying a bunch of different methods. """
	text = denoise_text(text)
	text = remove_special_characters(text)
	text = fix_contractions(text)
	return text

def random_downsampling(corpus, class_col = "rating", max_value = 300000):
	""" Reduces all instances of all classes to a certain maximum value.
	"""   
	
	corpus_1 = corpus[corpus[class_col] == 1.0]
	corpus_2 = corpus[corpus[class_col] == 2.0]
	corpus_3 = corpus[corpus[class_col] == 3.0]
	corpus_4 = corpus[corpus[class_col] == 4.0]
	corpus_5 = corpus[corpus[class_col] == 5.0]
	
	corpus_1 = corpus_1.sample(max_value)
	corpus_2 = corpus_2.sample(max_value)
	corpus_3 = corpus_3.sample(max_value)
	corpus_4 = corpus_4.sample(max_value)
	corpus_5 = corpus_5.sample(max_value)

	return pd.concat([corpus_1, corpus_2, corpus_3, corpus_4, corpus_5], axis=0)

# ======================= #
# neural network training #
# ======================= #

def flat_f1(true_labels, preds):
	""" Flattens predictions and labels and omputes macro f1-score.
	"""
	pred_flat = np.argmax(preds, axis=1).flatten()
	labels_flat = true_labels.flatten()
	return f1_score(labels_flat, pred_flat, average="macro")

def load_jsonl_to_df(path):
	""" Create dataframe from a JSON lines file. """
	data = []
	with open(path, 'r', encoding='utf-8') as f:
		for line in f:
			data.append(json.loads(line.rstrip('\n|\r')))
	df = pd.DataFrame(data)
	if type(df["review"][0]) == list:
		df["review"] = df.review.apply(" ".join)
	return df

def df_to_jsonl(df, filename, text_col="review", output_path="../corpora/splits/"):
	""" DataFrame with text column to Json Line Format. """

	df[text_col] = df.apply(lambda row: word_tokenize(row[text_col]), axis=1)
	df.to_json(f"{output_path}{filename}.json", orient='records', lines=True)

def format_time(elapsed):
	""" Takes a time in seconds and returns a string hh:mm:ss
	"""
	elapsed_rounded = int(round((elapsed)))
	return str(datetime.timedelta(seconds=elapsed_rounded))

def split_corpus(corpus, 
				 text_col = "review", 
				 label_col = "rating", 
				 split = 0.8,
				 output_path = "../corpora/splits/"):
	""" Splits corpus in Train, Val and Test set and saves them 
		as jsonl files.
	"""
	
	X_train, X_remain = train_test_split(corpus, 
										 train_size=split,
										 stratify=corpus[label_col])

	val_test_split = int((corpus.shape[0] * 0.2)/2)
	X_val = X_remain[:val_test_split]
	X_test = X_remain[val_test_split:]



	df_to_jsonl(X_train, "train", text_col = text_col, output_path = output_path)
	df_to_jsonl(X_val, "val")
	df_to_jsonl(X_test, "test")


def categorical_accuracy(preds, y):
	""" Returns accuracy per batch. """
	max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
	correct = max_preds.squeeze(1).eq(y)
	return correct.sum() / torch.FloatTensor([y.shape[0]])


def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs










