import numpy as np

from torchtext import data
from torchtext import datasets

from gensim.models import KeyedVectors


def getVectors(args, data):
	vectors = []

	if args.mode != 'rand':
		word2vec = KeyedVectors.load_word2vec_format('embeddings/GoogleNews-vectors-negative300.bin', binary=True)

		for i in range(len(data.TEXT.vocab)):
			word = data.TEXT.vocab.itos[i]
			if word in word2vec.vocab:
				vectors.append(word2vec[word])
			else:
				vectors.append(np.random.uniform(-0.01, 0.01, args.word_dim))
	else:
		for i in range(len(data.TEXT.vocab)):
			vectors.append(np.random.uniform(-0.01, 0.01, args.word_dim))

	return np.array(vectors)


class DATA():
	def __init__(self, args):
		
		self.TEXT = data.Field(batch_first=True, lower=True, fix_length=500)
		self.LABEL = data.LabelField()

		assigned_fields = {"review": ('text', self.TEXT), 
						   "rating": ('label', self.LABEL)}

		self.train_data, self.val_data, self.test_data = data.TabularDataset.splits(path="../corpora/splits/", 
																					train='train.json',
																					validation='val.json', 
																					test='test.json', 
																					format='json',
																					fields=assigned_fields,
																					skip_header = True)


		self.TEXT.build_vocab(self.train_data, self.test_data)
		self.LABEL.build_vocab(self.train_data)

		self.train_iter, self.test_iter = data.BucketIterator.splits((self.train_data, 
																	  self.test_data),
																	  batch_size=args.batch_size,
																	  sort=False,
																	  sort_key = lambda x: len(x.text),
																	  sort_within_batch=False,
																	  device=args.gpu)
		self.LABEL.build_vocab(self.train_data)

