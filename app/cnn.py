import argparse
import logging
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchtext import data, datasets

from utils import *



def main():

	# ================
	# time managment #
	# ================

	program_st = time.time()

	# =====================
	# cnn logging handler #
	# =====================
	logging_filename = f"../logs/cnn.log"
	logging.basicConfig(level=logging.INFO, filename=logging_filename, filemode="w")
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(levelname)s: %(message)s")
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)

	punctuation = ['!', '#','$','%','&', "'", '(',')','*', '+', ',', '-', '.', '/', 
				   ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', 
				   '{', '|', '}', '~', '`', '``']


	# =================
	# hyperparamaters # 
	# =================


	MAX_VOCAB_SIZE = args.max_features
	BATCH_SIZE = args.batch_size
	EPOCHS = args.epochs

	EMBEDDING_TYPE = args.embedding_type
	#TODO: ergänzen
	embeddings_dict = {"glove.6B.100d": "glove"}
	EMBEDDING_NAME = "unknown"
	if EMBEDDING_TYPE in embeddings_dict.keys():
		EMBEDDING_NAME = embeddings_dict[EMBEDDING_TYPE]


	corpus = pd.read_csv("../corpora/small_amazon_reviews_electronic.csv")

	# ===============
	# preprocessing #
	# ===============

	# stop_words = stopwords.words('english') + punctuation 

	REVIEW = data.Field(tokenize = "toktok",
						lower = True)

	RATING = data.LabelField()
	assigned_fields = {"review": ('text', REVIEW), 
					   "rating": ('label', RATING)}

	train_data, val_data, test_data = data.TabularDataset.splits(path="../corpora/splits/", 
																  train='train.json',
																  validation='val.json', 
																  test='test.json', 
																  format='json',
																  fields=assigned_fields,
																  skip_header = True)



	REVIEW.build_vocab(train_data, 
					   vectors = EMBEDDING_TYPE, 
					   unk_init = torch.Tensor.normal_,
					   max_size = MAX_VOCAB_SIZE)
	RATING.build_vocab(train_data)


	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	if torch.cuda.is_available():
		logging.info("GPU will be used.")


	train_iterator, val_iterator, test_iterator = data.BucketIterator.splits((train_data, val_data, test_data), 
																			 batch_size = BATCH_SIZE,
																			 device = device,
																			 sort_key = lambda x: len(x.text),
																			 sort = False,
																			 sort_within_batch=False)

	# ===========
	# CNN Model #
	# ===========

	class CNN(nn.Module):
		def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, 
					 output_dim, dropout, pad_idx):
			
			super().__init__()
			self.embedding = nn.Embedding(vocab_size, embedding_dim)
			self.convs = nn.ModuleList([
										nn.Conv2d(in_channels = 1, 
												  out_channels = n_filters, 
												  kernel_size = (fs, embedding_dim)) 
										for fs in filter_sizes
										])
			self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
			self.dropout = nn.Dropout(dropout)
			
		def forward(self, text):
			text = text.permute(1, 0)
			embedded = self.embedding(text)  
			embedded = embedded.unsqueeze(1)
			conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
			pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
			cat = self.dropout(torch.cat(pooled, dim = 1))
				
			return self.fc(cat)


	INPUT_DIM = len(REVIEW.vocab)
	EMBEDDING_DIM = 100
	N_FILTERS = 100
	FILTER_SIZES = [2,3,4]
	OUTPUT_DIM = len(RATING.vocab)
	DROPOUT = 0.5
	PAD_IDX = REVIEW.vocab.stoi[RATING.pad_token]

	model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, 
				OUTPUT_DIM, DROPOUT, PAD_IDX)

	OPTIMIZER = optim.Adam(model.parameters())
	CRITERION = nn.CrossEntropyLoss()


	# for pt model
	output_add = f'_e{EPOCHS}_bs{BATCH_SIZE}_mf{MAX_VOCAB_SIZE}_emb{EMBEDDING_NAME}'
	output_file = f'savefiles/cnnmodel{output_add}.pt'

	if args.load_savefile:
		model = torch.load(output_file)
	

	# load embeddings
	pretrained_embeddings = REVIEW.vocab.vectors 

	UNK_IDX = REVIEW.vocab.stoi[REVIEW.unk_token]
	model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
	model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

	# put model and loss criterion to device (cpu or gpu)
	model = model.to(device)
	CRITERION = CRITERION.to(device)


	# ================
	# train function #
	# ================

	def train(model, iterator, optimizer, criterion):
	
		epoch_loss = 0
		epoch_acc = 0
		
		model.train() 
		
		for batch in iterator:
			
			optimizer.zero_grad()
			predictions = model(batch.text)
			loss = criterion(predictions, batch.label)
			acc = categorical_accuracy(predictions, batch.label)
			loss.backward()
			optimizer.step()
			
			epoch_loss += loss.item()
			epoch_acc += acc.item()
			
		return epoch_loss / len(iterator), epoch_acc / len(iterator)


	# =====================
	# evaluation function #
	# =====================


	def evaluate(model, iterator, criterion):
	
		epoch_loss = 0
		epoch_acc = 0
		
		model.eval()
		
		with torch.no_grad():
			for batch in iterator:
				predictions = model(batch.text)
				loss = criterion(predictions, batch.label)
				acc = categorical_accuracy(predictions, batch.label)

				epoch_loss += loss.item()
				epoch_acc += acc.item()
			
		return epoch_loss / len(iterator), epoch_acc / len(iterator)

	# =================
	# actual training #
	# =================

	best_valid_loss = float('inf')


	for epoch in range(EPOCHS):

		start_time = time.time()
	  
		train_loss, train_acc = train(model, train_iterator, OPTIMIZER, CRITERION)
		valid_loss, valid_acc = evaluate(model, val_iterator, CRITERION)
		
		end_time = time.time()
		epoch_mins, epoch_secs = epoch_time(start_time, end_time)
		

		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			# TODO: string anpassen
			torch.save(model.state_dict(), output_file)
		
		logging.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
		logging.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
		logging.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

	# ============
	# Test model #
	# ============

	model.load_state_dict(torch.load(output_file))
	test_loss, test_acc = evaluate(model, test_iterator, CRITERION)

	logging.info(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="cnn", description="CNN for sentiment analysis.")
	parser.add_argument("--batch_size", "-bs", type=int, default=8, help="Indicates batch size.")
	parser.add_argument("--embedding_type", "-et", type=str, default="glove.6B.100d", help="Indicates embedding type.")
	parser.add_argument("--epochs", "-e", type=int, default=10, help="Indicates number of epochs.")
	#TODO: wird das genutzt?
	parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5, help="Set learning rate for optimizer.")
	parser.add_argument("--load_savefile", "-lsf", action="store_true", help="Loads savefile as input NN.")
	parser.add_argument("--max_features", "-mf", type=int, default=25000, help="Set the maximum size of vocabulary.")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	
	args = parser.parse_args()

	main()








