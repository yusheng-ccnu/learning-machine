import numpy as np

weight = [1,1,1]
xx=[1,-1,2,-3,4,-10,-2,-3,5,6,7,8,10,11,12,13,14,-4,-5,-6,-7]
yy=[5,1,10,5,26,82,2,5,37,50,65,82,102,125,140,157,206,11,15,13,28]
def f(x):
	return weight[0] * x**2 + weight[1] * x + weight[2]


def BGD(alpha = 0.0001,max_itor = 10000,epsilon=0.001):
	loss_1 = 0
	loss_2 = 0
	diff = 0
	itor = 0
	while max_itor > 0:
		for i in range(len(xx)):
			diff = f(xx[i]) - yy[i]
			weight[0] -= diff * alpha * xx[i] * xx[i]
			weight[1] -= diff * alpha * xx[i]
			weight[2] -= diff * alpha

		max_itor -= 1
		itor += 1
		print(" 迭代{}次",itor,"权重为:",weight)
		loss_1 = 0
		for j in range(len(xx)):
			loss_1 += f(xx[i]) ** 2
		if(loss_1 - loss_2 ) < epsilon:
			print("找到最佳权重",weight," 迭代{}次",itor)
			break
		else:
			loss_2 = loss_1
	return weight

BGD()
print(f(2))

