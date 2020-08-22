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
	def __init__(self, input_dim, output_dim, embedding_dim, 
				 embedding_type, n_filters, filter_sizes, 
				 dropout, transformer_model=""):
		super(KimCNN, self).__init__()

		self.in_channels = 1
		self.embedding_dim = embedding_dim

		if transformer_model:
			self.transformer_model = transformer_model
			self.embedding_dim = transformer_model.config.to_dict()['hidden_size']
		else:
			self.transformer_model = ""

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
		if self.transformer_model:
			with torch.no_grad():
				print("xsize1: ", x.size())
				x = self.transformer_model(x)[0]
				print("xsize2: ", x.size())
		else:
			x = x.permute(1, 0)
			x = self.embedding(x) 

		#print("\n bis hierhin \n") #todo 
		x = x.unsqueeze(1)
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

		if self.in_channels == 2:
			batch_size, _ = x.size()
			conv_in = self.embeddings(x).view(batch_size, 1, -1)
			x = self.embeddings(x).view(batch_size, 1, -1)
			x = torch.cat((conv_in, x), 1)
		
		x = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x]
		x = torch.cat(x, dim = 1)
		x = self.dropout(x)
		output = self.fc(x)
		return output

class DPCNN(nn.Module):
    """
    DPCNN for sentences classification.
    """
    def __init__(self, input_dim, output_dim, embedding_dim):
        super(DPCNN, self).__init__()
        self.channel_size = 250

        self.output_dim = output_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, embedding_dim), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(2*self.channel_size, self.output_dim)

    def forward(self, x):
        batch = x.shape[0]
        print("Batch: ", batch)
        print("size1: ", x.size())
        x = x.unsqueeze(1)
        print("size2: ", x.size())
        x = self.embedding(x)
        print("size3: ", x.size())

        # Region embedding
        x = self.conv_region_embedding(x)  
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size()[-2] > 2:
            x = self._block(x)

        print("size4: ", x.size())
        x = x.squeeze(2)
        x = x.squeeze(2)
        print("size4.2: ", x.size())
        x = x.view(batch, self.channel_size * 2)
        #x = x.view(self.output_dim)
        print("size5: ", x.size())
        x = self.linear_out(x)
        print("size6: ", x.size())
        return x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x

    """TODO: weg?
    def predict(self, x):
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        self.train(mode=True)
        return predict_labels
	"""

class KimCNN2(nn.Module):
	""" Code taken from here:
		https://towardsdatascience.com/identifying-hate-speech-with-bert-and-cnn-b7aa2cddd60d
	"""
	def __init__(self, embed_num, embed_dim, class_num, kernel_num, 
				 kernel_sizes, dropout, static, in_channels):

		super(KimCNN2, self).__init__()
		
		
		self.static = static
		self.embedding = nn.Embedding(embed_num, embed_dim)
		self.convs1 = nn.ModuleList([
				nn.Conv1d(in_channels, kernel_num, (size, embed_dim)) for size in kernel_sizes
			])
		self.dropout = nn.Dropout(dropout)
		self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num, class_num)
		

	def forward(self, x):
		if self.static:
			x = Variable(x)
		x = x.unsqueeze(1)
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] 
		x = torch.cat(x, 1)
		x = self.dropout(x) 
		output = self.fc1(x) 
		return output



class KimCNN3(nn.Module):
	""" Implementation taken from here:
		https://github.com/baaesh/CNN-sentence-classification-pytorch/blob/master/model.py
	"""
	def __init__(self, args, data, vectors):
		super(KimCNN3, self).__init__()

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

#TODO
"""
self.embed_num = embed_num # =vocab_size, maximum number of words in review
self.embed_dim = embed_dim # 768 for bert
self.class_num = class_num # 5 classes with reviews dataset
self.kernel_num = kernel_num # number of filters for each convolution operation
self.kernel_sizes = kernel_sizes # e.g. combinations of 2, 3, 4, ... words
self.in_channels = in_channels
"""
