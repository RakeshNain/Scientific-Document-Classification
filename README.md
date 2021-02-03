Scientific document classification is a key step for managing research articles and papers in forums like arxiv, Google Scholar and Microsoft Academic. In this project, we have given some abstracts crawled from arXiv, the task is to develop classification models which can make predictions and return the corresponding scientific fields of the source documents. Different from coarse grained classification tasks like sentiment analysis, this is a fine grained classification task where there are 100 filed classes in total.

# This package contains the Python code and output file 

Document Classification problem.

# Created by - Rakesh Nain

---Document Classification of abstracts based on a large training dataset using---

## Input Files

The required files are as follows:

|File name| Description |
|train_data_labels.csv|Training data contains train ids, abstracts and the labels

"train_data_labels.csv" can be downloaded from here https://drive.google.com/file/d/19hHYXgCsbHzYl0wkE6ZcYkcn_5qga2_G/view?usp=sharing


## Requirement to run the code

Python 3 is required to run the whole code.
The code can be run in the bash using the following command:
```bash
python Scientific Document Classification.py
```

## The files required:

The csv file that is train_data_labels.csv
And the Scientific Document Classification.py file

## Running Instructions:

The submitted file is a .py extension file, so it can be run using Pycharm.
The input files must be present in the same directory as the .py file or an absolute path must be given for train data.
**The code takes three hours to run.**
All the required libraries are imported at the start of the code.

## Input data format

1.	train_data_labels.csv
It contains three columns named – train_id, abstract and labels.
Training ids are just the indexes, so it is a number
Abstracts are crawled from arXiv and they are in the form of string
Labels are the classification of the abstracts in the form of strings
