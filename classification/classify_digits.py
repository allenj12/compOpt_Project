import neurolab as nl
import numpy as np
from classification import svm


def kernel(xm, xn):
    return (1 + np.dot(xm, xn)) ** 8


train_num = 500
net = nl.load("features/ann80.net")
print("Loaded NN")

all_data = np.loadtxt("combinedData.txt")
target = all_data[:, 0]
pixel_info = all_data[:, 1:]

train_info = pixel_info[:train_num]

test_info = pixel_info[train_num:]
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
clean_target = np.copy(target)
print("Using extracted features")
for i in range(0, 10):
    target_value = i
    target = np.copy(clean_target)
    target[target != target_value] = -1
    target[target == target_value] = 1
    train_target = target[:train_num]
    test_target = target[train_num:]

    #print(kernel(train_features, train_features.T).shape)
    #print np.array([train_target]).shape
    #print(np.dot(np.array([train_target]).T, np.array([train_target])).shape)

    alpha, b = svm.train_svm_kernel(train_features, train_target, C=0.07, K=kernel)

    e_test = svm.error(test_features, test_target, train_features, train_target, alpha, b, kernel)
    print("\tE_test for %d vs all = %f" % (target_value, e_test,))

print("Using base data")
for i in range(0, 10):
    target_value = i
    target = np.copy(clean_target)
    target[target != target_value] = -1
    target[target == target_value] = 1
    train_target = target[:train_num]
    test_target = target[train_num:]

    #print(kernel(train_features, train_features.T).shape)
    #print np.array([train_target]).shape
    #print(np.dot(np.array([train_target]).T, np.array([train_target])).shape)

    alpha, b = svm.train_svm_kernel(train_info, train_target, C=0.07, K=kernel)

    e_test = svm.error(test_info, test_target, train_info, train_target, alpha, b, kernel)
    print("\tE_test for %d vs all = %f" % (target_value, e_test,))
