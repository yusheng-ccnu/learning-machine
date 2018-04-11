import os
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
from sklearn.externals import joblib
#训练集
train_data = []
#训练集的value,即分类标签编号
y_train_data = []

test_data = []
y_test_data = []

#分类标签
label_index = {}



BASE_DIR = "./data"

path = os.path.join(BASE_DIR,"train_seg.txt")

#加载训练集
with open(path,'r',encoding="utf-8") as train_files:
	for line in train_files:
		words = line.split("\t")
		train_data.append(words[1])
		label = words[0]
		if label not in label_index:
			label_id = len(label_index)
			label_index[label] = label_id

		y_train_data.append(label_index[label])

#加载测试集
path = os.path.join(BASE_DIR,"test_seg.txt")
with open(path ,'r',encoding='utf-8') as test_files:
	for line in test_files:
		words = line.split("\t")
		test_data.append(words[1])
		label = words[0]
		y_test_data.append(label_index[label])

#将训练集转化为tfid特征向量

count_train = CountVectorizer(stop_words = None,min_df=0.0)
x_count_data = count_train.fit_transform(train_data)
tfid_trans = TfidfTransformer()
x_train_data = tfid_trans.fit(x_count_data).transform(x_count_data)

#将测试集也转化为tfid特征向量

count_test = CountVectorizer(stop_words = None,min_df=0.0,vocabulary=count_train.vocabulary_)
x_test_count = count_test.fit_transform(test_data)
tfid_test = TfidfTransformer()
x_test_data = tfid_test.fit(x_test_count).transform(x_test_count)
#print("######################训练贝叶斯模型#################")
#clf_bayes = MultinomialNB().fit(x_train_data,y_train_data)

#持久化训练的模型
#joblib.dump(clf_bayes,"bayes.m")
print(label_index)
clf_bayes = joblib.load("bayes.m")

question = ['谁 发现 的 南极洲']
x_question = count_test.fit_transform(question)
x_question_pre = tfid_test.fit(x_question).transform(x_question)
print(x_question_pre)
index = clf_bayes.predict(x_question_pre)[0]
for i in label_index.keys():
	if label_index[i] == index:
		print(i)

mnb_pred = clf_bayes.predict(x_test_data)
mnb_accuracy = metrics.precision_score(y_test_data, mnb_pred, average='weighted')
mnb_recall = metrics.recall_score(y_test_data, mnb_pred, average='weighted')
mnb_f1score = metrics.f1_score(y_test_data, mnb_pred, average='weighted')
print ('Naive Bayes Classifier accuracy:{0:.3f}'.format(mnb_accuracy))
print ('Naive Bayes Classifier recall:{0:.3f}'.format(mnb_recall))
print ('Naive Bayes Classifier f1 score:{0:.3f}'.format(mnb_f1score))
print ()

print("##################训练SVM##################")
clf_svm = SVC(gamma = 0.001, C=100.,kernel="linear")
clf_svm.fit(x_train_data,y_train_data)

svm_pred = clf_svm.predict(x_test_data)
svm_accuracy = metrics.precision_score(y_test_data, svm_pred, average='weighted')
svm_recall = metrics.recall_score(y_test_data, svm_pred, average='weighted')
svm_f1score = metrics.f1_score(y_test_data, svm_pred, average='weighted')
print ('svm Classifier accuracy:{0:.3f}'.format(svm_accuracy))
print ('svm Classifier recall:{0:.3f}'.format(svm_recall))
print ('svm Classifier f1 score:{0:.3f}'.format(svm_f1score))

