#!/usr/bin/env python
# Some parts of the code are from this tutorial: 
# https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX#scrollTo=6O_NbXFGMukX
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from keras.preprocessing.sequence import pad_sequences
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score 
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


import tensorflow as tf
import torch
torch.cuda.empty_cache()
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup

import sys
import random
import time
import utils

def main():

	# ================
	# time managment #
	# ================

	program_st = time.time()

	# =====================================
	# bert classification logging handler #
	# =====================================
	logging_filename = f"../logs/bert.log"
	logging.basicConfig(level=logging.INFO, filename=logging_filename, filemode="w")
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(levelname)s: %(message)s")
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)


	# =======================
	# predefined parameters #
	# =======================

	num_labels = 5

	batch_size = args.batch_size
	epochs = args.epochs
	learning_rate = args.learning_rate
	max_length = args.max_length

	model_name = "bert-base-uncased"

	class_name = "rating"
	text_name = "review"


	# ================
	# classification # 
	# ================


	# =======================
	# use GPU, if available #
	# =======================

	if torch.cuda.is_available(): 
		device = torch.device("cuda")  
		logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
		logging.info(f'Used GPU: {torch.cuda.get_device_name(0)}')
	else:
		logging.info('No GPU available, using the CPU instead.')
		device = torch.device("cpu")

	train_data = utils.load_jsonl_to_df('../corpora/splits/train.json')
	val_data = utils.load_jsonl_to_df('../corpora/splits/val.json')
	test_data = utils.load_jsonl_to_df('../corpora/splits/test.json')


	
	# tmp lists and result dicts #
	input_ids = []
	attention_masks = []

	val_input_ids = []
	val_attention_masks = []

	texts = train_data[text_name].values
	encoder = LabelEncoder()
	labels = encoder.fit_transform(train_data[class_name].values)

	X_val = val_data[text_name].values
	y_val = LabelEncoder().fit_transform(val_data[class_name].values)

	encoder_mapping = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))

	# ==============
	# tokenization #
	# ==============

	tokenizer = BertTokenizer.from_pretrained(model_name, 
											  do_lower_case=True)

	for sent in texts:
		encoded_dict = tokenizer.encode_plus(sent,
											 add_special_tokens = True,
											 max_length = args.max_length,
											 pad_to_max_length = True,
											 return_attention_mask = True, 
											 truncation=True,
											 return_tensors = 'pt')

		input_ids.append(encoded_dict['input_ids'])
		attention_masks.append(encoded_dict['attention_mask'])


	for sent in X_val:
		encoded_dict = tokenizer.encode_plus(sent,
											 add_special_tokens = True,
											 max_length = args.max_length,
											 pad_to_max_length = True,
											 return_attention_mask = True,
											 truncation=True, 
											 return_tensors = 'pt')

		val_input_ids.append(encoded_dict['input_ids'])
		
		val_attention_masks.append(encoded_dict['attention_mask'])


	input_ids = torch.cat(input_ids, dim=0)
	attention_masks = torch.cat(attention_masks, dim=0)
	labels = torch.tensor(labels)


	val_input_ids = torch.cat(val_input_ids, dim=0)
	val_attention_masks = torch.cat(val_attention_masks, dim=0)
	val_labels = torch.tensor(y_val)


	train_dataset = TensorDataset(input_ids, attention_masks, labels)
	val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)


	# ============
	# DataLoader #
	# ============

	train_dataloader = DataLoader(train_dataset,
								  sampler = RandomSampler(train_dataset),
								  batch_size = batch_size)

	val_dataloader = DataLoader(val_dataset, 
								sampler = SequentialSampler(val_dataset),
								batch_size = batch_size)

	# ======== #
	# Training #
	# ======== #
	model = BertForSequenceClassification.from_pretrained(model_name, 
														  num_labels = num_labels,
														  output_attentions = False,
														  output_hidden_states = False).cuda()

	optimizer = AdamW(model.parameters(),
					  lr=learning_rate,
					  eps=1e-8)

	total_steps = len(train_dataloader) * epochs

	scheduler = get_linear_schedule_with_warmup(optimizer, 
												num_warmup_steps = 0, 
												num_training_steps = total_steps)

	
	training_stats = []
	total_t0 = time.time()

	validation_losses = {}
	train_losses, val_losses = [], []

	for epoch_i in range(0, epochs):
		print("")
		print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
		print('Now Training.')

		t0 = time.time()
		total_train_loss = 0
		model.train()

		for step, batch in enumerate(train_dataloader):
			if step % 50 == 0 and not step == 0:
				elapsed = utils.format_time(time.time() - t0)
				print('Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

			b_input_ids = batch[0].to(device)
			b_input_mask = batch[1].to(device)
			b_labels = batch[2].to(device)

			model.zero_grad()        

			loss, logits = model(b_input_ids, 
								 token_type_ids=None, 
								 attention_mask=b_input_mask, 
								 labels=b_labels)

			total_train_loss += loss.item()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			scheduler.step()

		# average loss (all batches)
		avg_train_loss = total_train_loss / len(train_dataloader)
		train_losses.append(avg_train_loss)   
		training_time = utils.format_time(time.time() - t0)

		print("")
		print("  Average training loss: {0:.2f}".format(avg_train_loss))
		print("  Training epoch took: {:}".format(training_time))
		
		# ========== #
		# Validation #
		# ========== #

		print("")
		print("Now Validating.")

		t0 = time.time()
		model.eval()

		total_eval_accuracy = 0
		total_eval_loss = 0
		nb_eval_steps = 0

		for batch in val_dataloader:
			
			b_input_ids = batch[0].to(device)
			b_input_mask = batch[1].to(device)
			b_labels = batch[2].to(device)
			
			with torch.no_grad():        

				(loss, logits) = model(b_input_ids, 
									   token_type_ids=None, 
									   attention_mask=b_input_mask,
									   labels=b_labels)
				
			# validation loss.
			total_eval_loss += loss.item()

			# Move logits and labels to CPU
			logits = logits.detach().cpu().numpy()
			label_ids = b_labels.to('cpu').numpy()

			total_eval_accuracy += utils.flat_f1(label_ids, logits)
			

		# final validation accuracy / loss
		avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
		print("  Validation Accuracy: {0:.2f}".format(avg_val_accuracy))
		
		avg_val_loss = total_eval_loss / len(val_dataloader)
		val_losses.append(avg_val_loss)
		validation_time = utils.format_time(time.time() - t0)
		print("  Validation Loss: {0:.2f}".format(avg_val_loss))
		print("  Validation took: {:}".format(validation_time))

		
		training_stats.append({'epoch': epoch_i + 1,
							   'train_loss': avg_train_loss,
							   'val_loss': avg_val_loss,
							   'val_acc': avg_val_accuracy,
							   'train_time': training_time,
							   'val_time': validation_time})

		current_epoch = f"epoch{epoch_i + 1}"
		validation_losses[current_epoch] = avg_val_loss

		# ================
		# Early Stopping #
		# ================


		if utils.early_stopping(validation_losses, patience=2):
			logging.info(f"Stopping epoch run early (Epoch {epoch_i}).")
			break



	logging.info(train_losses)
	logging.info(val_losses)
	logging.info(type(train_losses))
	logging.info(type(val_losses))

	plt.plot(train_losses, label="Training loss")
	plt.plot(val_losses, label="Validation loss")
	plt.legend()
	plt.title("Losses")
	plt.savefig(f"../results/bert_loss_e{args.epochs}.png")

	logging.info(f"Training for {class_name} done.")
	logging.info("Training took {:} (h:mm:ss) \n".format(utils.format_time(time.time()-total_t0)))
	print("--------------------------------\n")

	# =========
	# Testing #
	# =========

	test_input_ids = []
	test_attention_masks = []

	X_test = test_data[text_name].values
	y_test = LabelEncoder().fit_transform(test_data[class_name].values)


	for sent in X_test:
		encoded_dict = tokenizer.encode_plus(sent,
											 add_special_tokens = True,
											 max_length = args.max_length,
											 pad_to_max_length = True,
											 return_attention_mask = True,
											 truncation=True, 
											 return_tensors = 'pt')

		test_input_ids.append(encoded_dict['input_ids'])
		
		test_attention_masks.append(encoded_dict['attention_mask'])

	test_input_ids = torch.cat(test_input_ids, dim=0)
	test_attention_masks = torch.cat(test_attention_masks, dim=0)
	labels = torch.tensor(y_test)

	prediction_data = TensorDataset(test_input_ids, test_attention_masks, labels)
	prediction_sampler = SequentialSampler(prediction_data)
	prediction_dataloader = DataLoader(prediction_data, 
									   sampler=prediction_sampler, 
									   batch_size=batch_size)

	model.eval()

	predictions, true_labels = [], []

	for batch in prediction_dataloader:
		# Add batch to GPU
		batch = tuple(t.to(device) for t in batch)
		  
		# Unpack the inputs from our dataloader
		b_input_ids, b_input_mask, b_labels = batch
		  
		
		with torch.no_grad():
			outputs = model(b_input_ids, 
							token_type_ids=None, 
							attention_mask=b_input_mask)

		logits = outputs[0]

		# Move logits and labels to CPU
		logits = logits.detach().cpu().numpy()
		label_ids = b_labels.to('cpu').numpy()
		  
		# Store predictions and true labels
		predictions.append(logits)
		true_labels.append(label_ids)


	

	flat_predictions = np.concatenate(predictions, axis=0)
	flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
	flat_true_labels = np.concatenate(true_labels, axis=0)

	
	test_score = f1_score(flat_true_labels, flat_predictions, average="macro")		
	classes = test_data[class_name].drop_duplicates().tolist()

	logging.info(f"Test score: {np.around(test_score, decimals=4)}")
	logging.info("Training took {:} (h:mm:ss)".format(utils.format_time(time.time()-total_t0)))
	print("________________________________")
	print("________________________________\n")


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="bertclf", description="Bert classifier.")
	parser.add_argument("--batch_size", "-bs", type=int, default=8, help="Indicates batch size.")
	parser.add_argument("--epochs", "-e", type=int, default=10, help="Indicates number of epochs.")
	parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5, help="Set learning rate for optimizer.")
	parser.add_argument("--max_length", "-ml", type=int, default=510, help="Indicates the maximum document length.")
	parser.add_argument("--patience", "-p", type=int, default=3, help="Indicates patience for early stopping.")
	parser.add_argument("--save_date", "-sd", action="store_true", help="Indicates if the creation date of the results should be saved.")
	
	args = parser.parse_args()

	main()