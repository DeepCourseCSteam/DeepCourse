import sys
import numpy as np
from sklearn.datasets import load_svmlight_file
#map 48-phones to 48-intager, return a dict
def label_to_class(map_file = "../data/MLDS_HW1_RELEASE_v1/phones/48_39.map"):
	D = {}
	cnt = 0
	with open(map_file, "r") as f:
		for line in f:
			label = line.strip().split()[0]
			D[label] = cnt
			cnt += 1
	return D

#combie the ark file and the lab file to an output svmlight file
def to_svmlight_file(ark_file = "../data/MLDS_HW1_RELEASE_v1/fbank/train.ark", lab_file = "../data/MLDS_HW1_RELEASE_v1/label/train.lab", output_file = "../data/train.dat"):
	ark_f = open(ark_file, "r")
	lab_f = open(lab_file, "r")
	output_f = open(output_file, "w")

	D = label_to_class()

	for line in ark_f:
		y = lab_f.readline().strip().split(",")[-1]
		output_f.write(D[y], end = " ")
		x = line.strip().split()
		for i in range(1 , len(x)):
			output_f.write(str(i)+":"+str(x[i]), end = " ")
		output_f.write("# "+str(x[0]))

	ark_f.close()
	lab_f.close()
	output_f.close()

#load the svmligh file to the inout of DNN
def load_data( svmlight_file = "../data/train.dat", input_dimension = 69, validation = True):
	(X, Y) = load_svmlight_file(svmlight_file)
	X = X.toarray() # array !!!!!! not matrix !!!!!!
	D = input_dimension


	if validation == True:# 20% for validation
		split = int(len(Y)*(0.8))+1
	else:
		split = len(Y)
	trainX = [np.reshape(x, (D, 1)) for x in X[0:split] ]
	trainY = [vectorize(int(y)) for y in Y[0:split]]
	train_data = list(zip(trainX, trainY))
	validX = [np.reshape(x, (D, 1)) for x in X[split:] ]
	validY = [vectorize(int(y)) for y in Y[split:]]
	valid_data = list(zip(validX, validY))
	return (train_data, valid_data)

#make y a (output_dimension, 1) vector
def vectorize(y, output_dimension = 48):
	v = np.zeros((output_dimension, 1))
	v[y] = 1.0
	return v

if __name__== "__main__":
	(train_data, valid_data) = load_data(svmlight_file = "../data/try.dat")
	print(train_data)
