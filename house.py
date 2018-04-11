import numpy as np

x = [[150,2],[200,90],[250,110],[300,150],[350,43],[400,4],[600,4]]
y = [645,745,845,945,1145,1540,1840]
weight = [0,0]
def h(x):
	return 1/(1+np)

def BGD(alpha = 0.0001,max_itor = 10000,epsilon=0.01):
	loss_1 = 0
	loss_2 = 0
	diff = [0,0]
	itor = 0
	while max_itor > 0:
		itor += 1
		max_itor -= 1
		for i in range(len(x)):
			diff[0] = h(x[i]) - y[i]
			print(diff[0])
			weight[0] -= alpha * diff[0]*x[i][0]
			weight[1] -= alpha * diff[0]*x[i][1]

		loss_1 = 0
		for j in range(len(x)):
			loss_1 += (h(x[j]) - y[j])**2/(2*len(x))
			if abs(loss_1 - loss_2) < epsilon:
				break
			else:
				loss_2 = loss_1
		print(weight)
		print(loss_1,"	",loss_2)
		print("跌代次数",itor)
	print(weight)
BGD()