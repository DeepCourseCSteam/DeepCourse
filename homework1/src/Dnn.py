import sys
import numpy as np
import random
import mnist_loader
import pickle
import math
#the sigmoid  function
def sigmoid(a):
	return 1.0/(1.0+np.exp(-a))

#the vectorized sigmoid function
vector_sigmoid = np.vectorize(sigmoid)

#the derivative sigmoid function
def derivative_sigmoid(a):
	return sigmoid(a)*(1-sigmoid(a))

#the derivative sigmoid function
vector_derivative_sigmoid = np.vectorize(derivative_sigmoid)

#load Dnn mode filel
def load_model(model_file = "../model/default.mod"):
	with open(model_file, "rb") as f:
		model = pickle.load(f)
	return model

#Dnn class
class Dnn:
	def __init__(self, size, drop_rate):
		self.drop_rate = drop_rate
		self.size = size# a list of size of neurons in every hidden layer, e.g.[2, 3, 5]
		self.layer_num = len(size)# number of layers
		self.bias = [np.random.randn(j, 1) for j in size[1:]]# bias ofevery layer
		self.weight = [np.random.randn(i, j) for (i, j) in zip(size[1:], size[:-1])]# the weight matrix of every layer-to-layer
		"""
		cnt = 0.0
		for B in self.bias:
			cnt += 1
			for i in range(len(B)):
				for j in range(len(B[i])):
					B[i][j] = (i+j+cnt)/1000
		for W in self.bias:
			cnt += 1
			for i in range(len(W)):
				for j in range(len(W[i])):
					W[i][j] = (i+j+cnt)/1000
		"""
	def predict(self, x):# return the output when given input
		a = x
		for (b, W) in zip(self.bias, self.weight):
			a = vector_sigmoid(np.dot(W, a) + b)# a = f(Wa+b)
		return a

	def train(self, train_data, epoch = 30, mini_batch_size = 10, eta = 3.0, valid_data = None, model_file = "../model/default.mod"):#stochastic gradient descent update
		if valid_data:
			max_accuracy = 0
		for e in range(epoch):
			random.shuffle(train_data)#shuffle first
			mini_B = [ train_data[offset:offset+mini_batch_size] for offset in range(0, len(train_data), mini_batch_size) ]
			for mini_batch in mini_B:#mini-batch update
				self.update(mini_batch, eta)#eta: learning rate
			if valid_data:
				accuracy = self.evaluate(valid_data)
				print "Epoch", e, ": L2 Loss = ", self.loss, ",Train Accuracy = ",self.evaluate(train_data), ", Valid Accuracy = ",accuracy
				if accuracy > max_accuracy:
					max_accuracy = accuracy
					with open(model_file, "wb") as f:
						pickle.dump(self, f)
			else:
				print "Epoch", e, ": Train Accuracy = ",self.evaluate(train_data)
	def update(self, batch, eta):#batch update
		nabla_b = [np.zeros(b.shape) for b in self.bias]#total delta b of all samples
		nabla_W = [np.zeros(W.shape) for W in self.weight]#total delta W of all samples
		for (x, y) in batch:
			(delta_nabla_b, delta_nabla_W) = self.backpropagate(x, y)
			nabla_b = [nb+dnb for (nb, dnb) in zip(nabla_b, delta_nabla_b)]
			nabla_W = [nW+dnW for (nW, dnW) in zip(nabla_W, delta_nabla_W)]
		self.weight = [W - (eta/float(len(batch)))*nW for (W, nW) in zip(self.weight, nabla_W)]#update W
		self.bias = [b - (eta/float(len(batch)))*nb for (b, nb) in zip(self.bias, nabla_b)]#update b

	def backpropagate(self, x, y):# back propagation to calculate the gradient
		nabla_b = [np.zeros(b.shape) for b in self.bias]
		nabla_W = [np.zeros(W.shape) for W in self.weight]
		A = [x]#the list of neurons in every layer
		Z = []#the list of input in every layer
		a = x

		#forward
		for (b, W, drop_rate) in zip(self.bias, self.weight, self.drop_rate):
			z = np.dot(W, a)+b # z = Wa+b
			Z.append(z)
			a = vector_sigmoid(z) # a = f(z)

			#perform dropout
			if drop_rate != 0:
				drop_list = np.random.choice(len(a), int(math.floor(len(a)*drop_rate)), replace=False)
				for i in drop_list:
					a[i] = 0.0
			A.append(a)
		#backward
		#refer to lecture to see the formula....
		delta = self.derivative_cost(A[-1], y)* vector_derivative_sigmoid(Z[-1]) # pC/pz elemnent-wise multiply
		
		self.loss = sum((A[-1]-y)**2);
		nabla_b[-1] = delta
		nabla_W[-1] = np.dot(delta, A[-2].transpose()) # (pz/wij).(pC/pz)
		for l in range(2, self.layer_num):
			z = Z[-l]
			delta = np.dot(self.weight[-l+1].transpose(), delta)* vector_derivative_sigmoid(z)
			nabla_b[-l] = delta
			if l == self.layer_num:
				nabla_W[-l] = np.dot(delta, Z[0])
			else:
				nabla_W[-l] = np.dot(delta, A[-l-1].transpose())

		return (nabla_b, nabla_W)

	def derivative_cost(self, output, y):# derivative of the L2 lost fuction
		return (output - y)

	def evaluate(self, test_data):#evaluate the accuracy on the test data
		score = 0
		test_result = [( np.argmax(self.predict(x)), np.argmax(y)) for (x, y) in test_data]
		score = sum([1 for (x, y) in test_result if x == y])
		return score / float(len(test_data))

if __name__ == "__main__":
	(train_data, valid_data, test_data)= mnist_loader.load_data_wrapper()
	dnn = Dnn([784, 10, 10], [0, 0, 0])
	dnn.train(list(train_data)[0:300], 30, 10, 3, valid_data = list(test_data))
	#dnn = load_model()
	#print(dnn.evaluate(list(test_data)))
