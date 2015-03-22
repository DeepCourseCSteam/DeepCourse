import sys
import numpy 
import Dnn
import Dataset 

if __name__ == "__main__":
    (train_data, valid_data) = Dataset.load_data(svmlight_file = "../data/small.dat", validation=True)
    #for (x, y) in train_data:
    #    print(x, " label: ", y)
    dnn = Dnn.Dnn([69, 128, 256, 128, 96, 48], [0, 0, 0.3, 0.5, 0])
    #print(train_data[0][1].shape)
    dnn.train(train_data, epoch=1000,eta=1, mini_batch_size=256, valid_data = valid_data, model_file = "../model/small.mod")
