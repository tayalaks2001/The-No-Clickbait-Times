# The-No-Clickbait-Times

## About the idea
Tired of clickbait-y and innacurate headlines, and long, boring articles in your news feed? The No-Clickbait Times is here to help. Using NLP, The No-Clickbait Times aims to analyse the headline and text, and summarise the article accordingly in a single line or two, giving an accurate description so your daily news feed can be precise and to the point.

## Requirements
numpy, pandas, matplotlib, seaborn

Scikit-Learn

gensim

nltk

GoogleNews-vectors-negative300.bin (training dataset, available online)

bert-extractive-summarizer

## Component details
Run Final_model.ipynb with the requirements present to get an output csv that you can sift through using ResultComparision.ipynb.

Index.html is our sample webpage for displaying how the results could look with a finished product.

ClickbaitClassifier.ipynb, bertSummarizer.ipynb and SummaryReviewer.ipynb are the individual components that make up Final_model.ipynb

train.csv (labeled) and test.csv (unlabeled) are datasets extracted from kaggle.com/micdsouz/news-clickbait that we clean, train and demonstrate our model on.
