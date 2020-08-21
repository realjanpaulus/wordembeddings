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


class SimpleCNN(nn.Module):
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



class KimCNN2(nn.Module):
	""" Code taken from here:
		https://github.com/galsang/CNN-sentence-classification-pytorch/blob/master/model.py
	"""
	def __init__(self, embed_num, embed_dim, class_num, kernel_num, 
				 kernel_sizes, dropout, static):
		super(KimCNN2, self).__init__()
		self.embed_num = embed_num
		self.embed_dim = embed_dim
		self.class_num = class_num
		self.kernel_num = kernel_num
		self.kernel_sizes = kernel_sizes
		
		self.static = static
		self.embed = nn.Embedding(embed_num, embed_dim)
		self.convs1 = nn.ModuleList([
				nn.Conv2d(1, kernel_num, (size, embed_dim)) for size in kernel_sizes
			])
		self.dropout = nn.Dropout(dropout)
		self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num, class_num)
		self.sigmoid = nn.Sigmoid()
		
	def forward(self, x):
		if self.static:
			x = Variable(x)
		x = x.unsqueeze(1)  # (N, Ci, W, embed_dim)
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, kernel_num, W), ...]*len(Ks)
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, kernel_num), ...]*len(Ks)
		x = torch.cat(x, 1)
		x = self.dropout(x)  # (N, len(Ks) * kernel_num)
		output = self.fc1(x)  # (N, class_num)
		return output



class KimCNN(nn.Module):
	""" Implementation taken from here:
		https://github.com/baaesh/CNN-sentence-classification-pytorch/blob/master/model.py
	"""
	def __init__(self, args, data, vectors):
		super(KimCNN, self).__init__()

		self.args = args

		self.word_emb = nn.Embedding(args.word_vocab_size, args.word_dim, padding_idx=1)
		# initialize word embedding with pretrained word2vec
		if args.mode != 'rand':
			self.word_emb.weight.data.copy_(torch.from_numpy(vectors))
		if args.mode in ('static', 'multichannel'):
			self.word_emb.weight.requires_grad = False
		if args.mode == 'multichannel':
			self.word_emb_multi = nn.Embedding(args.word_vocab_size, args.word_dim, padding_idx=1)
			self.word_emb_multi.weight.data.copy_(torch.from_numpy(vectors))
			self.in_channels = 2
		else:
			self.in_channels = 1

		# <unk> vectors is randomly initialized
		nn.init.uniform_(self.word_emb.weight.data[0], -0.05, 0.05)

		for filter_size in args.FILTER_SIZES:
			conv = nn.Conv1d(self.in_channels, args.num_feature_maps, args.word_dim * filter_size, stride=args.word_dim)
			setattr(self, 'conv_' + str(filter_size), conv)

		self.fc = nn.Linear(len(args.FILTER_SIZES) * 100, args.class_size)

	def forward(self, batch):
		x = batch.text
		batch_size, seq_len = x.size()

		conv_in = self.word_emb(x).view(batch_size, 1, -1)
		if self.args.mode == 'multichannel':
			conv_in_multi = self.word_emb_multi(x).view(batch_size, 1, -1)
			conv_in = torch.cat((conv_in, conv_in_multi), 1)

		conv_result = [
			F.max_pool1d(F.relu(getattr(self, 'conv_' + str(filter_size))(conv_in)), seq_len - filter_size + 1).view(-1,
																													self.args.num_feature_maps)
			for filter_size in self.args.FILTER_SIZES]

		out = torch.cat(conv_result, 1)
		out = F.dropout(out, p=self.args.dropout, training=self.training)
		out = self.fc(out)

		return out