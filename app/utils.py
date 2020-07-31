import pandas as pd

import re, string, unicodedata
import nltk
import contractions
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer

import gensim  
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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