# Basic Libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt # we only need pyplot
sb.set() # set the default Seaborn style for graphics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

newsData = pd.read_csv("train.csv")
testData = pd.read_csv("test.csv")
#removing all rows with unspecified label
newsData = newsData[newsData.label != 'other']
#Cleaning the remaining data
newsData = newsData.dropna(axis=0)
newsData.reset_index(drop=True, inplace=True)
#Cleaning test data
testData = testData[["title","text"]]
testData.dropna(axis=0, inplace=True)

X = newsData.title
y = newsData.label
for i in range(18330):
  if y[i]=='clickbait':
    y[i]=True
  else:
    y[i]=False

y = y.astype('bool')

clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,3), min_df=0.001)),
                ('clf', MultinomialNB())])
clf = clf.fit(X, y)
y_test_title_pred = clf.predict(testData.title)

X = newsData.text
clf = clf.fit(X, y)
y_test_text_pred = clf.predict(testData.text)

y_test_pred = pd.DataFrame(y_test_title_pred | y_test_text_pred)

#-----------------------------------------------------------------------------------------------------------------------

from gensim.models import KeyedVectors,Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
stopWords = set(stopwords.words("english"))
import nltk
import string

filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)

# To tokenize the string into words
def tokenizing(ss1,ss2):
    sentence1 = word_tokenize(ss1)
    sentence2 = word_tokenize(ss2)
    return sentence1,sentence2

# To clean the data : remove non alphanumeric characters
def cleaning(sentence1,sentence2):
    cleaned_ss1=[]
    for i in sentence1:
        if (i not in stopWords) and (i.isalnum()) :
            cleaned_ss1.append(i)

    cleaned_ss2=[]
    for i in sentence2:
        if i not in stopWords and i.isalnum():
            cleaned_ss2.append(i)
    return cleaned_ss1,cleaned_ss2

# the first function with a self defined formula to check if two sentences are related or similar
def check_spam(cleaned_ss1,cleaned_ss2):
    count = 0
    for i in cleaned_ss1:
        for j in cleaned_ss2:
            try:
                if model.similarity(i,j)>=0.3 :
                    count+=1
            except:
                pass
    if count>=(len(cleaned_ss1)*len(cleaned_ss2))**0.5:
        spam = False
    else :
        spam = True
    return spam

# Function that returns the norm(magnitude) of the vector
def magnitude(vector):
    sum_of_squares = 0
    k=0
    for num in vector:
        sum_of_squares += (num*num)
        k+=1
    result = sum_of_squares**0.5
    return result

# Function that returns the dot product of two vectors
def dot_product(u,v):
    dot = 0
    k=0
    for num in v:
        dot+=u[k]*v[k]
        k=k+1
    return dot

# Second function to check if two sentences are similar
def double_check(cleaned_ss1, cleaned_ss2):
    tolerance = 0.3
    v = [0]
    for i in cleaned_ss1:
        try:
            v = v + model[i]
        except:
            pass
    u = [0]
    for i in cleaned_ss2:
        try:
            u = u + model[i]
        except:
            pass

    if (magnitude(u) == 0 or magnitude(v) == 0):
        spam = True
        cosine = 0
    else:
        cosine = dot_product(u, v) / (magnitude(v) * magnitude(u))
        if (cosine > tolerance):
            spam = False
        else:
            spam = True
    return spam


# Third function to check if two sentences are similar or not
def triple_check(cleaned_ss1, cleaned_ss2):
    tolerance = 1.5
    v = [0]
    for i in cleaned_ss1:
        try:
            v = v + model[i]
        except:
            pass
    u = [0]
    for i in cleaned_ss2:
        try:
            u = u + model[i]
        except:
            pass
    diff = magnitude(u) - magnitude(v)

    if (-tolerance <= diff <= tolerance):
        spam = False
    else:
        spam = True
    return spam

from summarizer import Summarizer
model = Summarizer()

for i in range(testData.shape[0]):
    if y_test_pred[0][i]:
        orig_summary = model(testData['text'][i], ratio=0.05, min_length=6)
        orig_summary = ''.join(orig_summary)
        orig_headline = testData["title"][i]
        headline = orig_headline.lower()
        summary = orig_summary.lower()
        headline, summary = tokenizing(headline, summary)
        headline, summary = cleaning(headline, summary)
        check1 = check_spam(headline, summary)
        check2 = double_check(headline, summary)
        check3 = triple_check(headline, summary)
        if (check1 and check2) or (check2 and check3) or(check1 and check3):  # at least 2 are true
            testData["title"][i] = orig_summary

# testData now actually contains our result titles with their respective content
# saving this as a new csv,

testData.to_csv('result.csv', index=False)






