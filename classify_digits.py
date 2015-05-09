import neurolab as nl
import numpy as np
from classification import svm


def kernel(xm, xn):
    return (1 + np.dot(xm, xn)) ** 8


net = nl.load("features/ann80.net")
print("Loaded NN")

all_data = np.loadtxt("combinedData.txt")
target = all_data[:, 0]
target[target != 1] = -1
pixel_info = all_data[:, 1:]

train_target = target[:300]
train_info = pixel_info[:300]

test_target = target[7500:]
test_info = pixel_info[7500:]
print("Loaded info")

#cuts the last layer off of ann making the net that makes the features
inputParams = [[-1, 1]] * pixel_info.shape[1]
featureNet = nl.net.newff(inputParams, [80])
featureNet.layers[0].np['w'][:] = net.layers[0].np['w']
featureNet.layers[0].np['b'][:] = net.layers[0].np['b']
print("Chopped off output layer of NN")

train_features = featureNet.sim(train_info)
print("Simulated train info")
test_features = featureNet.sim(test_info)
print("Simulated test info")

#print(kernel(train_features, train_features.T).shape)
#print np.array([train_target]).shape
#print(np.dot(np.array([train_target]).T, np.array([train_target])).shape)

alpha, b = svm.train_svm_kernel(train_features, train_target, C=0.06, K=kernel)

print alpha
print b
