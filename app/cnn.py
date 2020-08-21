# Note: the following code is copied and adjusted from the following repo:
# https://github.com/baaesh/CNN-sentence-classification-pytorch

import argparse
import copy
import os
import torch

from gensim.models import KeyedVectors

from tensorboardX import SummaryWriter
from torch import nn, optim
from time import gmtime, strftime

from models import KimCNN
from data import DATA, getVectors


def train(args, data, vectors):
	""" Train function for CNN.
	"""
	if args.model == "kimcnn":
		model = KimCNN(args, data, vectors)
	model.to(torch.device(args.device))

	parameters = filter(lambda p: p.requires_grad, model.parameters())
	optimizer = optim.Adadelta(parameters, lr=args.learning_rate)
	criterion = nn.CrossEntropyLoss()

	writer = SummaryWriter(log_dir='../logs/' + args.model_time)

	model.train()
	acc, loss, size, last_epoch = 0, 0, 0, -1
	max_test_acc = 0

	print_summary = False
	iterator = data.train_iter
	for i, batch in enumerate(iterator):

		present_epoch = int(iterator.epoch)
		if present_epoch == args.epoch:
			break
		if present_epoch > last_epoch:
			print('epoch:', present_epoch + 1)
			print_summary = True
		last_epoch = present_epoch

		pred = model(batch)

		optimizer.zero_grad()
		batch_loss = criterion(pred, batch.label)
		loss += batch_loss.item()
		batch_loss.backward()
		nn.utils.clip_grad_norm_(parameters, max_norm=args.norm_limit)
		optimizer.step()

		_, pred = pred.max(dim=1)
		acc += (pred == batch.label).sum().float()
		size += len(pred)

		if print_summary:
			print_summary = False
			acc /= size
			acc = acc.cpu().item()
			test_loss, test_acc = test(model, data)
			c = present_epoch

			writer.add_scalar('loss/train', loss, c)
			writer.add_scalar('acc/train', acc, c)
			writer.add_scalar('loss/test', test_loss, c)
			writer.add_scalar('acc/test', test_acc, c)

			print(f'train loss: {loss:.3f} / test loss: {test_loss:.3f}'
				  f' / train acc: {acc:.3f} / test acc: {test_acc:.3f}')

			if test_acc > max_test_acc:
				max_test_acc = test_acc
				best_model = copy.deepcopy(model)

			acc, loss, size = 0, 0, 0
			model.train()

	writer.close()
	print(f'max test acc: {max_test_acc:.3f}')

	return best_model



def test(model, data, mode='test'):
	if mode == 'dev':
		iterator = iter(data.dev_iter)
	else:
		iterator = iter(data.test_iter)

	criterion = nn.CrossEntropyLoss()
	model.eval()
	acc, loss, size = 0, 0, 0

	for batch in iterator:
		pred = model(batch)

		batch_loss = criterion(pred, batch.label)
		loss += batch_loss.item()

		_, pred = pred.max(dim=1)
		acc += (pred == batch.label).sum().float()
		size += len(pred)

	acc /= size
	acc = acc.cpu().item()
	return loss, acc


def load_model(args, data):
	model = KimCNN(args, data)
	model.load_state_dict(torch.load(args.model_path))

	if args.gpu > -1:
		model.cuda(args.gpu)

	return model


def main():

	print(f"Loading the '{args.dataset}' dataset.")

	data = DATA(args)
	""" TODO: weiter
	TEXT = data.Field(batch_first=True, lower=True, fix_length=500)
	LABEL = data.LabelField()

	assigned_fields = {"review": ('text', TEXT),
					   "rating": ('label', LABEL)}
	"""



	vectors = getVectors(args, data)

	setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
	setattr(args, 'class_size', len(data.LABEL.vocab))
	setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
	setattr(args, 'FILTER_SIZES', [3, 4, 5])

	if args.gpu > -1:
		setattr(args, 'device', "cuda:0")
	else:
		setattr(args, 'device', "cpu")

	print('Start Training!')
	best_model = train(args, data, vectors)

	torch.save(best_model.state_dict(), f'savefiles/CNN_KimCNN_{args.dataset}_{args.model_time}.pt')

	print('Training finished!')


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--batch-size', default=8, type=int)
	parser.add_argument('--dataset', default='Amazon Reviews', help="Dataset name.")
	parser.add_argument('--dropout', default=0.5, type=float)
	parser.add_argument("--embedding_type", "-et", default="word2vec")
	parser.add_argument('--epoch', "-e", default=1, type=int)
	parser.add_argument('--gpu', default=0, type=int, help="")
	parser.add_argument('--learning-rate', default=0.1, type=float)
	parser.add_argument("--mode", "-md", default="non-static", help="available models: rand, static, non-static, multichannel")
	parser.add_argument("--model", "-m", default="kimcnn", help="Indicates used cnn model.")
	parser.add_argument('--norm-limit', default=3.0, type=float)
	parser.add_argument('--num-feature-maps', default=100, type=int)
	parser.add_argument('--word-dim', default=300, type=int)

	args = parser.parse_args()

	main()