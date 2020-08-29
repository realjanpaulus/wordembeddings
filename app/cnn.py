CUDA_LAUNCH_BLOCKING="1"
import argparse
import logging
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchtext import data, datasets

import transformers
import models
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

	BATCH_SIZE = args.batch_size
	DATA_PATH = args.datapath
	DROPOUT = 0.5
	EPOCHS = args.epochs

	FILTER_SIZES = [3,4,5]
	LEARNING_RATE = args.learning_rate
	MAX_FEATURES = args.max_features 
	N_FILTERS = 100
	
	
	# ============
	# embeddings #
	# ============

	EMBEDDING_TYPE = args.embedding_type
	
	if EMBEDDING_TYPE == "fasttext-en":
		EMBEDDING_NAME = "fasttext.en.300d"
		EMBEDDING_DIM = 300
	elif EMBEDDING_TYPE == "fasttext-simple":
		EMBEDDING_NAME = "fasttext.simple.300d"
		EMBEDDING_DIM = 300	
	elif EMBEDDING_TYPE == "glove-840":
		EMBEDDING_NAME = "glove.840B.300d"
		EMBEDDING_DIM = 300
	elif EMBEDDING_TYPE == "glove-6":
		EMBEDDING_NAME = "glove.6B.300d"
		EMBEDDING_DIM = 300
	elif EMBEDDING_TYPE == "glove-twitter":
		EMBEDDING_NAME = "glove.twitter.27B.200d"
		EMBEDDING_DIM = 200
	else:	
		EMBEDDING_NAME = "unknown"
		EMBEDDING_DIM = 300

	# ===============
	# preprocessing #
	# ===============

		
	TEXT = data.Field(tokenize = "toktok",
						  lower = True)


	LABEL = data.LabelField(dtype = torch.long)
	assigned_fields = {"review": ('text', TEXT), 
					   "rating": ('label', LABEL)}
	
	train_data, val_data, test_data = data.TabularDataset.splits(path=DATA_PATH, 
																 train=f'train{args.splitnumber}.json',
																 validation=f'val{args.splitnumber}.json', 
																 test=f'test{args.splitnumber}.json', 
																 format='json',
																 fields=assigned_fields,
																 skip_header = True)

	
	TEXT.build_vocab(train_data, 
					 vectors = EMBEDDING_NAME, 
					 unk_init = torch.Tensor.normal_,
					 max_size = MAX_FEATURES)
	LABEL.build_vocab(train_data)

	INPUT_DIM = len(TEXT.vocab)
	OUTPUT_DIM = len(LABEL.vocab)

	
	if torch.cuda.is_available():       
		device = torch.device("cuda")
		logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
		logging.info(f'Device name: {torch.cuda.get_device_name(0)}')
	else:
		logging.info('No GPU available, using the CPU instead.')
		device = torch.device("cpu")


	train_iterator, val_iterator, test_iterator = data.BucketIterator.splits((train_data, val_data, test_data), 
																			 batch_size = BATCH_SIZE,
																			 device = device,
																			 sort_key = lambda x: len(x.text),
																			 sort = False,
																			 sort_within_batch=False)

	
	
	# ===========
	# CNN Model #
	# ===========

		
	print("\n")
	logging.info("#####################################")
	logging.info(f"Input dimension (= vocab size): {INPUT_DIM}")
	logging.info(f"Output dimension (= n classes): {OUTPUT_DIM}")
	logging.info(f"Embedding dimension: {EMBEDDING_DIM}")
	logging.info(f"Embedding type: {EMBEDDING_TYPE}")
	logging.info(f"Number of filters: {N_FILTERS}")
	logging.info(f"Filter sizes: {FILTER_SIZES}")
	logging.info(f"Dropout: {DROPOUT}")
	logging.info("#####################################")
	print("\n")

	
	if args.model == "kimcnn":
		model = models.KimCNN(input_dim = INPUT_DIM,
							  output_dim = OUTPUT_DIM, 
							  embedding_dim = EMBEDDING_DIM, 
							  embedding_type = EMBEDDING_TYPE,
							  n_filters = N_FILTERS, 
							  filter_sizes = FILTER_SIZES, 
							  dropout = DROPOUT)

		OPTIMIZER = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)
		CRITERION = nn.CrossEntropyLoss()
	else:
		logging.info(f"Model '{args.model}' does not exist. Script will be stopped.")
		exit()


	# for pt model
	output_add = f'_bs{BATCH_SIZE}_mf{MAX_FEATURES}_{EMBEDDING_TYPE}'
	output_file = f'savefiles/cnnmodel{output_add}.pt'

	if args.load_savefile:
		model.load_state_dict(torch.load(output_file))
	

	# load embeddings
	pretrained_embeddings = TEXT.vocab.vectors 
	UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
	PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

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


	def evaluate(model, iterator, criterion, return_lists=False):
	
		epoch_loss = 0
		epoch_acc = 0
		
		model.eval()

		if return_lists:
			pred_labels, true_labels = [], []
		
		with torch.no_grad():
			for batch in iterator:
				predictions = model(batch.text)
				loss = criterion(predictions, batch.label)
				acc = categorical_accuracy(predictions, batch.label)

				epoch_loss += loss.item()
				epoch_acc += acc.item()

				if return_lists:
					predictions = predictions.detach().cpu().numpy()
					batch_labels = batch.label.to('cpu').numpy()
					pred_labels.append(predictions)
					true_labels.append(batch_labels)
		
		if return_lists:
			return epoch_loss / len(iterator), epoch_acc / len(iterator), pred_labels, true_labels
		else:
			return epoch_loss / len(iterator), epoch_acc / len(iterator)

	# =================
	# actual training #
	# =================

	best_val_loss = float('inf')

	train_losses = []
	val_losses = []
	val_losses_epochs = {}
	total_train_time = time.time()

	for epoch in range(EPOCHS):

		start_time = time.time()
	  
		train_loss, train_acc = train(model, train_iterator, OPTIMIZER, CRITERION)
		val_loss, val_acc = evaluate(model, val_iterator, CRITERION)

		train_losses.append(train_loss)
		val_losses.append(val_loss)
		val_losses_epochs[f"epoch{epoch+1}"] = val_loss
		
		end_time = time.time()
		epoch_mins, epoch_secs = epoch_time(start_time, end_time)
		

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			torch.save(model.state_dict(), output_file)
		
		logging.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
		logging.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
		logging.info(f'\tVal. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%')


		if early_stopping(val_losses_epochs, patience=args.patience):
			logging.info(f"Stopping epoch run early (Epoch {epoch}).")
			break


	logging.info("Training took {:} (h:mm:ss) \n".format(format_time(time.time()-total_train_time)))
	print("--------------------------------\n")

	plt.plot(train_losses, label="Training loss")
	plt.plot(val_losses, label="Validation loss")
	plt.legend()
	plt.title(f"Losses (until epoch {epoch})")
	plt.savefig(f"../results/{args.model}_loss_{args.embedding_type}_{args.splitnumber}_bs{args.batch_size}_mf{args.max_features}_lr{args.learning_rate}.png")

	# ============
	# Test model #
	# ============

	total_test_time = time.time()
	model.load_state_dict(torch.load(output_file))

	if args.save_confusion_matrices:
		test_loss, test_acc, pred_labels, true_labels = evaluate(model, test_iterator, CRITERION, return_lists=True)

		flat_predictions = np.concatenate(pred_labels, axis=0)
		flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
		flat_true_labels = np.concatenate(true_labels, axis=0)


		logging.info("Saving confusion matrices.")
		testd_ = load_jsonl_to_df(f"{args.datapath}/test{args.splitnumber}.json")
		classes = testd_["rating"].drop_duplicates().tolist()

		cm_df = pd.DataFrame(confusion_matrix(flat_true_labels, flat_predictions), index=classes, columns=classes)
		cm_df.to_csv(f"../results/cm_{args.embedding_type}_{args.splitnumber}_bs{args.batch_size}_mf{args.max_features}_lr{args.learning_rate}.csv")

	else:
		test_loss, test_acc = evaluate(model, test_iterator, CRITERION)


	test_output = f'\nTest Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%'
	test_outputfile = f"../results/{args.embedding_type}_{args.splitnumber}_bs{args.batch_size}_mf{args.max_features}_lr{args.learning_rate}.txt"

	with open(test_outputfile, "w") as txtfile:
		txtfile.write(f"Last epoch: {epoch}{test_output}")

	logging.info(test_output)
	logging.info("Testing took {:} (h:mm:ss) \n".format(format_time(time.time()-total_test_time)))
	print("--------------------------------\n")
	logging.info("Total duration {:} (h:mm:ss) \n".format(format_time(time.time()-program_st)))
	



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="cnn", description="CNN for sentiment analysis.")
	parser.add_argument("--batch_size", "-bs", type=int, default=50, help="Indicates batch size.")
	parser.add_argument("--datapath", "-dp", default="../corpora/splits/", help="Indicates dataset path.")
	parser.add_argument("--embedding_type", "-et", type=str, default="glove-6", help="Indicates embedding type. \
		Possible values: 'fasttext-en', 'fasttext-simple', 'glove-840', 'glove-6', 'glove-twitter'.")
	parser.add_argument("--epochs", "-e", type=int, default=500, help="Indicates number of epochs.")
	parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Set learning rate for optimizer.")
	parser.add_argument("--load_savefile", "-lsf", action="store_true", help="Loads savefile as input NN.")
	parser.add_argument("--max_features", "-mf", type=int, default=25000, help="Set the maximum size of vocabulary.")
	parser.add_argument("--model", "-m", default="kimcnn", help="Indicates used cnn model: Available: 'kimcnn'.")
	parser.add_argument("--patience", "-p", type=int, default=3, help="Indicates patience for early stopping.")
	parser.add_argument("--save_confusion_matrices", "-scm", action="store_true", help="Indicates if confusion matrices should be saved." )
	parser.add_argument("--splitnumber", "-sn", type=int, default=1, help="Indicates split number, e.g. train2.")

	args = parser.parse_args()

	main()








