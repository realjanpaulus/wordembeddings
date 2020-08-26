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
#from torchtext import data, datasets
from torch.autograd import Variable


class KimCNN(nn.Module):
	""" Implementation of the KimCNN Model of this paper:
		https://www.aclweb.org/anthology/D14-1181.pdf
	"""
	def __init__(self, input_dim, output_dim, embedding_dim, 
				 embedding_type, n_filters, filter_sizes, 
				 dropout):
		super(KimCNN, self).__init__()

		self.in_channels = 1
		self.embedding_dim = embedding_dim
		self.embedding = nn.Embedding(input_dim, self.embedding_dim, padding_idx=1)
		self.embedding.weight.requires_grad = False
		self.convs = nn.ModuleList([
				nn.Conv1d(in_channels = self.in_channels,
				  		  out_channels = n_filters, 
					  	  kernel_size = (fs, embedding_dim)) 
				for fs in filter_sizes
			])
		self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
		self.dropout = nn.Dropout(dropout)
	
	
	def forward(self, x):
		x = x.permute(1, 0)
		x = self.embedding(x) 
		x = x.unsqueeze(1)
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
		x = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x]
		x = torch.cat(x, dim = 1)
		x = self.dropout(x)
		output = self.fc(x)
		return output

class KimCNN2(nn.Module):
	""" Implementation of the KimCNN Model of this paper:
		https://www.aclweb.org/anthology/D14-1181.pdf
	"""
	def __init__(self, input_dim, output_dim, embedding_dim, 
				 embedding_type, n_filters, filter_sizes, 
				 dropout):
		super(KimCNN2, self).__init__()

		self.in_channels = 1
		self.embedding_dim = embedding_dim
		self.embedding = nn.Embedding(input_dim, self.embedding_dim, padding_idx=1)
		self.embedding.weight.requires_grad = False
		self.convs = nn.ModuleList([
				nn.Conv1d(in_channels = self.in_channels,
				  		  out_channels = n_filters, 
					  	  kernel_size = (fs, embedding_dim)) 
				for fs in filter_sizes
			])
		self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
		self.dropout = nn.Dropout(dropout)
	
	
	def forward(self, x):
		x = x.permute(1, 0)
		x = self.embedding(x) 
		x = x.unsqueeze(1)
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
		x = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x]
		x = torch.cat(x, dim = 1)
		x = self.dropout(x)
		output = self.fc(x)
		return output
