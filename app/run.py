import argparse
from itertools import product
import logging
import subprocess
import sys
import time

def main():

	program_st = time.time()


	logging.basicConfig(level=logging.DEBUG, 
						filename=f"../logs/run.log", 
						filemode="w")
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(levelname)s: %(message)s")
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)
	

	logging.info("Starting optimization.")
	batch_sizes = [50, 128]
	learning_rates = [0.01, 0.001]
	if args.embedding_type == "fasttext":
		embeddings = ['fasttext-en', 'fasttext-simple'] 
	elif args.embedding_type == "glove":
		embeddings = ['glove-840', 'glove-6', 'glove-twitter']
	else:
		logging.info(f"Embedding type '{args.embedding_type}' is unknown.")
		exit()
	max_features = [25000, 50000]
	split_numbers = [1, 2, 3]

	cartesian_inputs = list(product(batch_sizes, 
									learning_rates, 
									embeddings, 
									max_features,
									split_numbers))

	
	for idx, t in enumerate(cartesian_inputs):
		print("--------------------------------------------")
		logging.info(f"Argument combination {idx+1}/{len(cartesian_inputs)}.")
		logging.info(f"Batch size: {t[0]}.")
		logging.info(f"Learning rate: {t[1]}.")
		logging.info(f"Embeddings: {t[2]}.")
		logging.info(f"Max features: {t[3]}.")
		logging.info(f"Split number: {t[4]}.")
		print("--------------------------------------------")

		command = f"python cnn.py -bs {t[0]} -lr {t[1]} -et {t[2]} -mf {t[3]} -sn {t[4]} -e {args.epochs}"
		subprocess.call(["bash", "-c", command])
		print("\n")
	program_duration = float(time.time() - program_st)
	logging.info(f"Overall run-time: {int(program_duration)/60} minute(s).")

	
	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="run", description="Runs cnn script with multiple arguments.")
	parser.add_argument("--embedding_type", "-et", type=str, default="fasttext", help="Indicates embedding type. Possible values: 'fasttext', 'glove'.")
	parser.add_argument("--epochs", "-e", type=int, default=500, help="Indicates number of epochs.")
	args = parser.parse_args()

	main()