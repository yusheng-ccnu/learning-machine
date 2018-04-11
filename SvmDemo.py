from sklearn import svm

x = [[2,0],[1,1],[2,3]]
y = [0,0,1]

clf = svm.SVC(kernel="linear")
clf.fit(x,y)

print(clf)

#获取支持向量
print(clf.support_vectors_)

#获取支持向量的下标
print(clf.support_)

#在类别里面分别找到的向量个数，比如下面的输出是[1,1]
#则表示在0分类上的支持向量的个数是1,1分类的支持向量的
#个数也是1
print(clf.n_support_)

xx = [2.,0.]
print(clf.predict([xx]))
