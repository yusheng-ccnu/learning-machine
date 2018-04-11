from sklearn import svm
import numpy as np
import pylab as pl

#创建四十个点
np.random.seed(1)
X = np.r_[np.random.randn(20,2) - [2,2],np.random.randn(20,2) + [2,2]]
Y = [0] * 20 + [1] * 20

print(Y)

clf = svm.SVC(kernel = "linear")
clf.fit(X,Y)

w = clf.coef_[0]

print(w)

a = -w[0]/w[1]
xx = np.linspace(-5,5)
print("a=",a)
print(xx)

yy = a * xx - (clf.intercept_[0])/w[1]

b = clf.support_vectors_[0]
yy_down = a*xx + (b[1] - a*b[0])

b = clf.support_vectors_[-1]
yy_up = a*xx + (b[1] - a*b[0])

pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

print("support vector=",clf.support_vectors_)

pl.scatter(X[:,0],X[:,1],c=Y,cmap=pl.cm.Paired)
pl.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=80,facecolors="none",marker="o",edgecolors='g',color="")

pl.axis("tight")
pl.show()

	