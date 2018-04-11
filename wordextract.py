from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import os


BASE_PATH = "./data"
print("load files")
test = []
with open("37915") as testfile:
	for line in testfile:
		test.append(line)
print(len(test))

count_1 = CountVectorizer()
x_test = count_1.fit_transform(test)
print(count_1.get_feature_names())
print(x_test.toarray())