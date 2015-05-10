import neurolab as nl
import numpy as np
from classification import svm
import functools
from multiprocessing import Pool, cpu_count
import random


def kernel(xm, xn):
    return (1 + np.dot(xm, xn)) ** 8

random.seed()
train_num = 500
net = nl.load("features/ann80.net")
print("Loaded NN")

all_data = np.loadtxt("combinedData.txt")
#index_list = range(0, all_data.shape[0])
#train_list = random.sample(index_list, train_num)
#test_list = [x for x in index_list if x not in train_list]
target = all_data[:, 0]
pixel_info = all_data[:, 1:]

train_info = pixel_info[:train_num]
print train_info.shape

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
num_processes = cpu_count()
print("Number of processes = %s" % (num_processes,))
pool = Pool(processes=num_processes)
for i in range(0, 10):
    target_value = i
    target = np.copy(clean_target)
    target[target != target_value] = -1
    target[target == target_value] = 1
    train_target = target[:train_num]
    test_target = target[train_num:]

    C_range = np.arange(0.01, 1.01, 0.05)
    all_errors = pool.map(functools.partial(svm.ecv, train_features, train_target, kernel), C_range)
    min_error = min(all_errors)
    c_index = all_errors.index(min_error)
    C = C_range[c_index]

    #print(kernel(train_features, train_features.T).shape)
    #print np.array([train_target]).shape
    #print(np.dot(np.array([train_target]).T, np.array([train_target])).shape)

    alpha, b = svm.train_svm_kernel(train_features, train_target, C=C, K=kernel)

    e_test = svm.error(test_features, test_target, train_features, train_target, alpha, b, kernel)
    print("\tE_test for %d vs all = %f" % (target_value, e_test,)),
    print("\tC=%f" % (C,))

print("Using base data")
for i in range(0, 10):
    target_value = i
    target = np.copy(clean_target)
    target[target != target_value] = -1
    target[target == target_value] = 1
    train_target = target[:train_num]
    test_target = target[train_num:]

    C_range = np.arange(0.01, 1.01, 0.05)
    all_errors = pool.map(functools.partial(svm.ecv, train_info, train_target, kernel), C_range)
    min_error = min(all_errors)
    c_index = all_errors.index(min_error)
    C = C_range[c_index]

    #print(kernel(train_features, train_features.T).shape)
    #print np.array([train_target]).shape
    #print(np.dot(np.array([train_target]).T, np.array([train_target])).shape)

    alpha, b = svm.train_svm_kernel(train_info, train_target, C=C, K=kernel)

    e_test = svm.error(test_info, test_target, train_info, train_target, alpha, b, kernel)
    print("\tE_test for %d vs all = %f" % (target_value, e_test,)),
    print("\tC=%f" % (C,))
