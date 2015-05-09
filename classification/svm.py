import numpy as np
from cvxopt import solvers
from cvxopt import matrix


def construct_Q(X):
    x = X[0]
    d = x.shape[0]
    Q = np.zeros((d + 1, d + 1))
    Q[1:, 1:] = np.identity(d)
    return Q


def construct_p(X):
    x = X[0]
    d = x.shape[0]
    p = np.zeros(d + 1)
    return p


def construct_A(X, Y):
    x = X[0]
    N = Y.shape[0]
    d = x.shape[0]
    A = np.zeros((N, d + 1))
    A[:, 0] = Y
    count = 0
    for x, y in zip(X, Y):
        A[count, 1:] = y * x
        count += 1
    return A


def construct_c(y):
    N = y.shape[0]
    c = np.ones(N)
    return c


def train_svm(X, y):
    Q = matrix(construct_Q(X))
    p = matrix(construct_p(X))
    A = matrix(-1 * construct_A(X, y))
    c = matrix(-1 * construct_c(y))
    x = solvers.qp(P=Q, q=p, G=A, h=c)['x']
    u = np.array(x)[:, 0]
    return u


def construct_G(X, y, K):
    N = y.shape[0]
    G = np.zeros((N, N))
    for n in range(0, N):
        for m in range(0, N):
            G[n, m] = y[n] * y[m] * K(X[n], X[m])
    return G


def construct_alpha_constraints(y, C):
    N = y.shape[0]
    coefficients = np.zeros((2 * N, N))
    coefficients[:N, :] = np.identity(N)
    coefficients[N:, :] = -1.0 * np.identity(N)
    constraints = np.zeros((2 * N, 1))
    constraints[:N, :] = C
    return coefficients, constraints


def train_svm_kernel(X, y, C, K):
    solvers.options['show_progress'] = False
    N = y.shape[0]
    G = matrix(construct_G(X, y, K))
    ones = matrix(-1.0, (N, 1))
    Y = matrix(y)
    coefficients, constraints = construct_alpha_constraints(y, C)
    coeff = matrix(coefficients)
    constr = matrix(constraints)
    #print("G.size = %r" % (G.size,))
    #print("ones.size = %r" % (ones.size,))
    #print("coeff.size = %r" % (coeff.size,))
    #print("constr.size = %r" % (constr.size,))
    #print("Y.T.size = %r" % (Y.T.size,))
    a = solvers.qp(P=G, q=ones, G=coeff, h=constr, A=Y.T, b=matrix(0.0))['x']
    alpha =  np.array(a)
    s = -1
    for i in range(0, N):
        if alpha[i] < C and alpha[i] > 0:
            s = i
            break
    if s == -1:
        print("No s found")
    summation = 0
    for n in range(0, N):
        if alpha[n] > 0:
            summation += (alpha[n] * y[n] * K(X[n], X[s]))
    b = y[s] - summation
    return alpha, b


def error(X, y, X_train, y_train, alpha, b, K):
    e = 0.0
    d = alpha.shape[0]
    N = y.shape[0]
    alpha_mask = alpha <= 0
    alpha_masked = np.ma.MaskedArray(alpha, alpha_mask)
    summation = np.zeros((X.shape[0]))
    for n in range(0, d):
        summation += np.ma.dot(np.multiply(y_train, K(X_train, X)), alpha_masked)
    summation += b
    err = np.logical_not(np.equal(np.sign(summation), y))
    e = np.sum(err)
    final_e = e / N
    return final_e
