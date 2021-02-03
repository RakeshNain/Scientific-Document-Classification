# SCIENTIFIC DOCUMENT CLASSIFICATION
# Created By: Rakesh Nain

# Importing the libraries required for the task
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Loading the train dataset
train = pd.read_csv("train_data_labels.csv")

# Making a dataframe with not null abstracts of train dataset
train = train[pd.notnull(train['abstract'])]

# Taking the required columns from the dataframe
col = ['label', 'abstract']
train = train[col]
# Creating label_id column which is the numeric value for the categorical data in label
train['label_id'] = train['label'].factorize()[0]

# Printing the columns in the train dataframe
print(train.columns)

# Creating a dictionary for label and label id
label_df = train[['label', 'label_id']].drop_duplicates().sort_values('label_id')
label_dict = dict(label_df[['label_id', 'label']].values)

# train-test split
train, test = train_test_split(train, test_size=0.2)

# Plotting to check the class balance
fig = plt.figure(figsize=(8,6))
train.groupby('label').abstract.count().plot.bar(ylim=0)

# uncomment below line of code if you want to visualise class imbalance in our data
# plt.show()

# Data Preprocessing

# Creating a list of all the stopwords
stopword_list = stopwords.words('english')

# Creating class for lemmatizing
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        # Replacing the unnecessary characters
        doc = re.sub(r'(/(){}\[\]\|@,;#+_)', ' ', doc)
        # Making tokens for the words
        tokenizer = RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(doc)
        # Finding the numbers to remove from the tokens
        nums = RegexpTokenizer(r"\d+").tokenize(doc)
        # Removing the digits from the tokens list
        tokens = [w for w in tokens if w not in nums]
        # Removing the stop words and the tokens of length 1
        final_tokens = [x for x in tokens if x not in stopword_list and len(x) >1]
        # Returning the list of tokens after lemmatising
        return [self.wnl.lemmatize(t) for t in final_tokens]

# Feature Extraction

# Calculating Term Frequency - Inverse Document Frequency
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True,  ngram_range=(1, 1), analyzer='word', input='content', lowercase=True, max_df=0.95, min_df=0.001, tokenizer=LemmaTokenizer())

# Fitting tfidf_vectorizer and returning document-term matrix
abstract_train = tfidf_vectorizer.fit_transform(train['abstract'])
label_train = np.asarray(train['label_id'])
abstract_test = tfidf_vectorizer.transform(test['abstract'])

# Fitting and Predicting on the best model
model = SVC(C=1, kernel='sigmoid', decision_function_shape='ovo')
model.fit(abstract_train.toarray(), label_train)
label_predict = model.predict(abstract_test.toarray())

print("Accuracy of our model is: ", accuracy_score(test["label_id"], label_predict))

