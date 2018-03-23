# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import metrics


filepath = unicode('20news-bydate-train','utf-8')
rawData = datasets.load_files(filepath,encoding="latin1")

count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(rawData.data)

tfidf_transformer = TfidfTransformer()
tfid = tfidf_transformer.fit_transform(x_train_counts)

clf = MultinomialNB().fit(tfid,rawData.target)


test_clf = Pipeline([
	('vect',CountVectorizer()),
	('tfid',TfidfTransformer()),
	('clf',MultinomialNB()),
	])
test_clf.fit(rawData.data , rawData.target)
testData = datasets.load_files("20news-bydate-test",encoding="latin1")
predicted = test_clf.predict(testData.data)
result = metrics.classification_report(testData.target,predicted,target_names = testData.target_names)
print(result)

