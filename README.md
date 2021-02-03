# wordembeddings
In a series of experiments, an English Amazon review corpus was used to examined how **sentiment analysis** with *CNNs + word embeddings* performs in comparison to other models such as *BERT*, *logistic regression* or *SVM*.


## Structure

- **app**. This directory contains the scripts `cnn.py`, `bert.py`, `ml.py`, `run.py`, `run_bert.py` used for classification and optimization as well as the helper script `utils.py` and the file `models.py`, which contains the KimCNN architecture. The notebook `processing_and_results.ipynb` was used for corpus preprocessung and to summarize the classification results.
- **corpora**. This directory contains the Amazon User Review Corpus and the splitted corpora for the cross validation.
- **results**. This directory contains the results of the classification experiments as text files as well as figures of the confusion matrices and the train- and validation loss of the CNN and BERT models.
