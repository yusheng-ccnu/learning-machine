import numpy as np


# 训练集  
# 每个样本点有3个分量 (x0,x1,x2)  
x = [(1, 0., 3), (1, 1., 3), (1, 2., 3), (1, 3., 2), (1, 4., 4)]  
# y[i] 样本点对应的输出  
y = [95.364, 97.217205, 75.195834, 60.105519, 49.342380]  

weight = [0,0,0]





def h(x):
	return weight[0]*x[0] + weight[1]*x[1] + weight[2]*x[2]
 
def BGD():
	#学习率
	alpha = 0.01
	#初始梯度
	diff=[0,0]
	#最大迭代次数1000
	max_itor = 10000000
	itor = 0
	loss_1 = 0
	loss_2 = 0
	while max_itor > 0:
		max_itor -= 1
		itor += 1
		for i in range(len(x)):
 			diff[0] = h(x[i]) - y[i]
 			weight[0] -= alpha * diff[0]*x[i][0]
 			weight[1] -= alpha * diff[0]*x[i][1]
 			weight[2] -= alpha * diff[0]*x[i][2]

		loss_1 = 0
		for j in range(len(x)):
 			loss_1 += (h(x[j])-y[j])**2/(len(x)*2)
		if abs(loss_1 - loss_2) < 0.001:
 			break
		else:
 			loss_2 = loss_1
 			
		#print("loss_1=",loss_1," loss_2=",loss_2)
		print("last weight",weight)
	print("总的跌代次数",itor)

BGD()

import matplotlib.pyplot as plt
plt.plot([1,2,3,4],[1,4,9,16],'ro')
plt.ylabel('some numbers')
#plt.show()
 		