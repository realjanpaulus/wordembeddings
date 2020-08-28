# wordembeddings
Files for the course "Word Embeddings organisieren" at the Julius-Maximilians-University, SS20.


## Structure

- **app**. This directory contains the scripts `cnn.py`, `bert.py`, `ml.py`, `run.py`, `run_bert.py` used for classification and optimization as well as the helper script `utils.py` and the file `models.py`, which contains the KimCNN architecture. The notebook `processing_and_results.ipynb` was used for corpus preprocessung and to summarize the classification results.
- **corpora**. This directory contains the Amazon User Review Corpus and the splitted corpora for the cross validation.
- **results**. This directory contains the results of the classification experiments as text files as well as figures of the confusion matrices and the train- and validation loss of the CNN and BERT models.