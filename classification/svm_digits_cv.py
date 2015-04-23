import numpy as np
import svm
import random
import sys
from scipy import signal
from matplotlib import pyplot as plt
from multiprocessing import Pool
import functools
import time


def K(xm, xn):
    kernel = (1 + np.dot(xm, xn)) ** 8
    return kernel


def sign(x):
    if x > 0:
        return 1.0
    else:
        return -1.0


def error(X, y, X_train, y_train, alpha, b):
    e = 0.0
    d = alpha.shape[0]
    N = y.shape[0]
    for i, x in enumerate(X):
        summation = 0
        for n in range(0, d):
            if alpha[n] > 0:
                summation += (alpha[n] * y_train[n] * K(X_train[n], x))
        summation += b
        if sign(summation) != y[i]:
            e += 1.0
    final_e = e / N
    return final_e


def ecv(X, Y, C):
    print("Starting C = %r" % (C,))
    start = time.clock()
    N = Y.shape[0]
    err = 0.0
    for i in range(0, N):
        x_out = X[i]
        y_out = Y[i]
        X_cv = np.concatenate((X[:i], X[i + 1:]))
        Y_cv = np.concatenate((Y[:i], Y[i + 1:]))
        alpha, b = svm.train_svm_kernel(X_cv, Y_cv, C, K)
        d = alpha.shape[0]
        summation = 0
        for n in range(0, d):
            if alpha[n] > 0:
                summation += (alpha[n] * Y_cv[n] * K(X_cv[n], x_out))
        summation += b
        if sign(summation) != y_out:
            err += 1.0
    end = time.clock()
    elapsed = end - start
    print("C = %r finished in %r seconds" % (C, elapsed,))
    return err / N


def classify(pt, alpha, b, X_train, Y_train):
    d = alpha.shape[0]
    summation = 0
    for n in range(0, d):
        if alpha[n] > 0:
            summation += (alpha[n] * Y_train[n] * K(X_train[n], pt))
    summation += b
    return summation


train_file = "ZipDigits.train.txt"
train_data = open(train_file, 'r')

test_file = "ZipDigits.test.txt"
test_data = open(test_file, 'r')

images = []
tests = []
random.seed()
for line in train_data:
    values = line.split()
    digit = int(float(values[0]))
    values.pop(0)
    img = np.zeros(len(values))
    for i, val in enumerate(values):
        img[i] = float(val)
    img = np.reshape(img, (16, 16))
    ones = np.ones(img.shape)
    img += ones
    images.append((digit, img))

for line in test_data:
    values = line.split()
    digit = int(float(values[0]))
    values.pop(0)
    img = np.zeros(len(values))
    for i, val in enumerate(values):
        img[i] = float(val)
    img = np.reshape(img, (16, 16))
    ones = np.ones(img.shape)
    img += ones
    images.append((digit, img))

features_other = []
features_one = []
features = []
min_x1 = sys.maxint
max_x1 = -sys.maxint - 1
min_x2 = sys.maxint
max_x2 = -sys.maxint - 1
dim = 8
for digit in images:
    num = digit[0]
    image = digit[1]
    sobel_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    sobel_y = np.array([[1,  2,   1],
                       [0,  0,   0],
                       [-1, -2, -1]])
    g_x = signal.convolve2d(image, sobel_x)
    g_y = signal.convolve2d(image, sobel_y)
    g = np.hypot(g_x, g_y)
    grad_mean = np.mean(g)
    if grad_mean > max_x2:
        max_x2 = grad_mean
    elif grad_mean < min_x2:
        min_x2 = grad_mean

    norm = np.mean(image)
    if norm > max_x1:
        max_x1 = norm
    elif norm < min_x1:
        min_x1 = norm

    x = np.array([norm, grad_mean])
    if num == 1:
        features.append((1.0, x))
    else:
        features.append((-1.0, x))

shift_x1 = -(min_x1 + max_x1) / 2.0
scale_x1 = 1.0 / (max_x1 + shift_x1)

shift_x2 = -(min_x2 + max_x2) / 2.0
scale_x2 = 1.0 / (max_x2 + shift_x2)

positivesx = []
positivesy = []
negativesx = []
negativesy = []
for i, ft in enumerate(features):
    classification = ft[0]
    x = ft[1]
    x1 = x[0]
    x2 = x[1]
    x1_normalized = scale_x1 * (x1 + shift_x1)
    x2_normalized = scale_x2 * (x2 + shift_x2)
    x_normalized = np.array([x1_normalized, x2_normalized])
    ft_normalized = (classification, x_normalized)
    features[i] = ft_normalized
    if classification == 1:
        features_one.append(x_normalized)
        positivesx.append(x1_normalized)
        positivesy.append(x2_normalized)
    else:
        features_other.append(x_normalized)
        negativesx.append(x1_normalized)
        negativesy.append(x2_normalized)


samples = random.sample(range(0, len(features)), 300)
training_set = []
for i in sorted(samples, reverse=True):
    ft = features.pop(i)
    training_set.append(ft)
test_set = features

X = [pt[1] for pt in training_set]
X = np.array(X)
Y = [pt[0] for pt in training_set]
Y = np.array(Y)

X_test = [pt[1] for pt in test_set]
X_test = np.array(X_test)
Y_test = [pt[0] for pt in test_set]
Y_test = np.array(Y_test)

C_range = np.arange(0.01, 1.01, 0.05)
min_ecv = 1.0
count = 0
start = time.clock()
pool = Pool(processes=8)
all_errors = pool.map(functools.partial(ecv, X, Y), np.arange(0.01, 1.01, 0.05))
min_error = min(all_errors)
c_index = all_errors.index(min_error)
end = time.clock()
elapsed = end - start
print("Overall elapsed time is %r seconds" % (elapsed,))
#for c in C_range:
#    if count % 10 == 0:
#        print count
#    count += 1
#    e_cv = ecv(X, Y, c)
#    if e_cv < min_ecv:
#        min_ecv = e_cv
C = C_range[c_index]
alpha, b = svm.train_svm_kernel(X, Y, C, K)
print("C = %r" % (C,))
print("Ein = %r" % (error(X, Y, X, Y, alpha, b),))
print("Etest = %r" % (error(X_test, Y_test, X, Y, alpha, b),))
plt_range = np.arange(-1.0, 1.0, 0.01)
resultx = []
resulty = []
epsilon = 0.01
print("Start timer")
start = time.clock()
for x in plt_range:
    for y in plt_range:
        pt = np.array([x, y])
        classification = classify(pt, alpha, b, X, Y)
        if classification >= -1.0 * epsilon and classification <= epsilon:
            resultx.append(x)
            resulty.append(y)
end = time.clock()
print("Elapsed time: %r" % (end - start,))
plt.plot(positivesx, positivesy, 'bo')
plt.plot(negativesx, negativesy, 'rx')
plt.plot(resultx, resulty)
plt.show()
