import argparse
from itertools import product
import logging
import subprocess
import sys
import time

def main():

	program_st = time.time()


	logging.basicConfig(level=logging.DEBUG, 
						filename=f"../logs/run_bert.log", 
						filemode="w")
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(levelname)s: %(message)s")
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)
	

	logging.info("Starting optimization.")
	learning_rates = [2e-5, 3e-5, 5e-5]
	split_numbers = [1, 2, 3]

	cartesian_inputs = list(product(learning_rates, split_numbers))

	
	for idx, t in enumerate(cartesian_inputs):
		print("--------------------------------------------")
		logging.info(f"Argument combination {idx+1}/{len(cartesian_inputs)}.")
		logging.info(f"Learning rate: {t[0]}.")
		logging.info(f"Split number: {t[1]}.")
		print("--------------------------------------------")

		command = f"python bert.py -lr {t[0]} -sn {t[1]}"

		if args.save_confusion_matrices:
			command += " -scm"

		subprocess.call(["bash", "-c", command])
		print("\n")
	program_duration = float(time.time() - program_st)
	logging.info(f"Overall run-time: {int(program_duration)/60} minute(s).")

	
	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(prog="run_bert", description="Runs bert script with multiple arguments.")
	parser.add_argument("--epochs", "-e", type=int, default=10, help="Indicates number of epochs.")
	parser.add_argument("--save_confusion_matrices", "-scm", action="store_true", help="Indicates if confusion matrices should be saved." )
	args = parser.parse_args()

	main()