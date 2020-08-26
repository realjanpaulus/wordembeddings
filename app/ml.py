#!/usr/bin/env python
import argparse
from datetime import datetime
import logging
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


import sys
import time

from utils import *


def main():

	# ================
	# time managment #
	# ================

	program_st = time.time()

	# =======================
	# predefined parameters #
	# =======================

	n_jobs = args.n_jobs
	cv = 3
	vectorizer = TfidfVectorizer(lowercase=False,
								 stop_words=None)

	
	corpus = pd.read_csv("../corpora/alter_small_amazon_reviews_electronic.csv")

	if args.testing:
		corpus = corpus.sample(100)

	features = corpus["review"]
	labels = corpus["rating"]


	# ================================
	# classification logging handler #
	# ================================
	logging_filename = f"../logs/svm.log"
	logging.basicConfig(level=logging.DEBUG, filename=logging_filename, filemode="w")
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(levelname)s: %(message)s")
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)

	

	# ============
	# Linear SVM #
	# ============

	st = time.time()
	logging.info(f"Starting the training of {args.model}.")
	if args.model == "svm":
		pipe = Pipeline(steps=[("vect", vectorizer),
							   ("clf", LinearSVC())])
		
		parameters = {"vect__ngram_range": [(1,1), (1,2), (2,3)],
					  "vect__max_features": [25000, 50000],
					  "clf__penalty": ["l2"],
					  "clf__loss": ["squared_hinge"],
					  "clf__tol": [0.01, 0.001],
					  "clf__C": [1, 2, 3],
					  "clf__max_iter": [1000, 3000, 5000]}

		if args.testing:
			parameters = {"vect__ngram_range": [(1,1)],
						  "vect__max_features": [100],
						  "clf__penalty": ["l2"],
						  "clf__loss": ["squared_hinge"],
						  "clf__tol": [0.01],
						  "clf__C": [1],
						  "clf__max_iter": [1000]}
		

		grid = GridSearchCV(pipe,
							parameters,
							cv=cv,
							error_score=0.0,
							n_jobs=args.n_jobs,
							scoring="accuracy")


		cv_results = cross_validate(grid, 
									features, 
									labels,
									cv=cv, 
									n_jobs=args.n_jobs,
									return_estimator=False,
									scoring="accuracy")


		acc = np.mean(cv_results['test_score'])
		logging.info(f"Accuracy: {acc}")

		with open("../results/svm.txt", "w") as txtfile:
			txtfile.write(f"Mean Accuracy: {acc}")
	elif args.model == "lr":
		pipe = Pipeline(steps=[("vect", vectorizer),
							   ("clf", LogisticRegression())])
		
		parameters = {"vect__ngram_range": [(1,1), (1,2), (2,3)],
					  "clf__penalty": ["l2"],
					  "clf__tol": [0.01, 0.001],
					  "clf__C": [1, 2, 3],
					  "clf__solver": ["liblinear"],
					  "clf__max_iter": [1000, 3000, 5000]}

		if args.testing:
			parameters = {"vect__ngram_range": [(1,1)],
						  "clf__penalty": ["l2"],
						  "clf__tol": [0.01],
						  "clf__C": [1],
						  "clf__solver": ["liblinear"],
						  "clf__max_iter": [1000]}
		

		grid = GridSearchCV(pipe,
							parameters,
							cv=cv,
							error_score=0.0,
							n_jobs=args.n_jobs,
							scoring="accuracy")


		cv_results = cross_validate(grid, 
									features, 
									labels,
									cv=cv, 
									n_jobs=args.n_jobs,
									return_estimator=False,
									scoring="accuracy")


		acc = np.mean(cv_results['test_score'])
		logging.info(f"Accuracy: {acc}")

		with open("../results/lr.txt", "w") as txtfile:
			txtfile.write(f"Mean Accuracy: {acc}")


	duration = float(time.time() - st)
	logging.info(f"Run-time {args.model}: {np.around(duration, decimals=4)} seconds")

	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="svm", description="Classification of LSVM.")
	parser.add_argument("--model", "-m", type=str, default="svm", help="Indicates ML Model. Possible values: lr, svm.")
	parser.add_argument("--n_jobs", "-nj", type=int, default=1, 
						help="Indicates the number of processors used for computation.")
	parser.add_argument("--testing", "-t", action="store_true", 
						help="Starts testing mode with a small subset of the corpus \
						and less tunable parameters.")
	args = parser.parse_args()

	main()
