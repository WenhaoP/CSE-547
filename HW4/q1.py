import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def svm_loss(w, b, C, D):
    X, y = D
    return (np.linalg.norm(w) ** 2) / 2 + C * np.sum(np.maximum(0, 1 - y * (X @ w + b)))

def BGD_converge_criterion(w_last, w_curr, b_last, b_curr, C, eps, D):
    loss_last = svm_loss(w_last, b_last, C, D)
    loss_curr = svm_loss(w_curr, b_curr, C, D)

    return (np.abs(loss_last - loss_curr) / loss_last * 100) < eps, loss_curr

def SGD_converge_criterion(delta_last, w_last, w_curr, b_last, b_curr, C, eps, D):
    loss_last = svm_loss(w_last, b_last, C, D)
    loss_curr = svm_loss(w_curr, b_curr, C, D)

    delta_curr = (delta_last + (np.abs(loss_last - loss_curr) / loss_last * 100)) / 2
    
    return delta_curr < eps, delta_curr, loss_curr

def MBGD_converge_criterion(delta_last, w_last, w_curr, b_last, b_curr, C, eps, D):
    loss_last = svm_loss(w_last, b_last, C, D)
    loss_curr = svm_loss(w_curr, b_curr, C, D)

    delta_curr = (delta_last + (np.abs(loss_last - loss_curr) / loss_last * 100)) / 2
    
    return delta_curr < eps, delta_curr, loss_curr

def partial_w(w, b, C, D):
    X, y = D
    ind = ((y * (X @ w + b)) < 1)
    return w + C * np.sum((-y * X.T) * ind, axis = 1)

def partial_b(w, b, C, D):
    X, y = D
    ind = ((y * (X @ w + b)) < 1)
    return C * np.sum(-y * ind)

def BGD():

    features = np.loadtxt("hw4-bundle/bundle_files/svm/data/features.txt", delimiter=",")
    target = np.loadtxt("hw4-bundle/bundle_files/svm/data/target.txt", delimiter=",")
    
    ### Algorithm hyperparameters ###
    n, d = features.shape
    beta = n
    eta = 3e-7
    eps = 0.25
    C = 100

    ### Algorithm starts ###
    start = time.perf_counter()

    t = 0
    k = 0
    B = (features.copy(), target.copy())

    # initializing parameters
    w_last = np.zeros(d)
    w_curr = np.zeros(d)
    b_last = 0
    b_curr = 0

    criterion_flag, loss_init = BGD_converge_criterion(w_last, w_curr, b_last, b_curr, C, eps, B)
    criterion_flag = False
    BGD_loss_history = [loss_init]

    print(f"At iteration {0}, the loss is {loss_init}")

    while (not criterion_flag):
        t += 1

        w_curr = w_last - eta * partial_w(w_last, b_last, C, B)
        b_curr = b_last - eta * partial_b(w_last, b_last, C, B)

        criterion_flag, loss_curr = BGD_converge_criterion(w_last, w_curr, b_last, b_curr, C, eps, B)
        BGD_loss_history.append(loss_curr)

        w_last = w_curr
        b_last = b_curr

        if (t % 10 == 0):
            print(f"At iteration {t}, the loss is {loss_curr}")

    end = time.perf_counter()

    print(f"After {t} iterations, the BGD algorithm converges with the final loss {loss_curr} and total runtime {end - start}.")
        
def SGD():

    features = np.loadtxt("hw4-bundle/bundle_files/svm/data/features.txt", delimiter=",")
    target = np.loadtxt("hw4-bundle/bundle_files/svm/data/target.txt", delimiter=",")

    ### Algorithm hyperparameters ###
    n, d = features.shape
    beta = 1
    eta = 1e-4
    eps = 1e-3
    C = 100

    rng = np.random.default_rng(1)
    permuted_idx = rng.permutation(np.arange(n))
    X_permuted = features[permuted_idx]
    y_permuted = target[permuted_idx]

    ### Algorithm starts ###
    start = time.perf_counter()
    t = 0
    k = 0

    # initializing parameters
    w_last = np.zeros(d)
    w_curr = np.zeros(d)
    b_last = 0
    b_curr = 0

    criterion_flag, delta_curr, loss_init = SGD_converge_criterion(0, w_last, w_curr, b_last, b_curr, C, eps, (X_permuted, y_permuted))
    criterion_flag = False
    SGD_loss_history = [loss_init]

    print(f"At iteration {0}, the loss is {loss_init}")

    while (not criterion_flag):
        t += 1

        B = (X_permuted[[k]], y_permuted[k])

        w_curr = w_last - eta * partial_w(w_last, b_last, C, B)
        b_curr = b_last - eta * partial_b(w_last, b_last, C, B)
        k = (k + 1) % n

        criterion_flag, delta_curr, loss_curr = SGD_converge_criterion(delta_curr, w_last, w_curr, b_last, b_curr, C, eps, (X_permuted, y_permuted))
        SGD_loss_history.append(loss_curr)

        w_last = w_curr
        b_last = b_curr

        if (t % 10 == 0):
            print(f"At iteration {t}, the loss is {loss_curr}")

    end = time.perf_counter()

    print(f"After {t} iterations, the SGD algorithm converges with the final loss {loss_curr} and total runtime {end - start}.")
        
def MBGD():

    features = np.loadtxt("hw4-bundle/bundle_files/svm/data/features.txt", delimiter=",")
    target = np.loadtxt("hw4-bundle/bundle_files/svm/data/target.txt", delimiter=",")

    ### Algorithm hyperparameters ###
    n, d = features.shape
    beta = 20
    num_batch = np.ceil(n / beta)
    eta = 1e-5
    eps = 1e-2
    C = 100

    rng = np.random.default_rng(547)
    permuted_idx = rng.permutation(np.arange(n))
    X_permuted = features[permuted_idx]
    y_permuted = target[permuted_idx]

    ### Algorithm starts ###
    start = time.perf_counter()
    t = 0
    k = 0

    # initializing parameters
    w_last = np.zeros(d)
    w_curr = np.zeros(d)
    b_last = 0
    b_curr = 0

    criterion_flag, delta_curr, loss_init = MBGD_converge_criterion(0, w_last, w_curr, b_last, b_curr, C, eps, (X_permuted, y_permuted))
    criterion_flag = False
    MBGD_loss_history = [loss_init]

    print(f"At iteration {0}, the loss is {loss_init}")

    while (not criterion_flag):
        t += 1

        B = (X_permuted[(beta * k): min(beta * (k + 1), n), ], y_permuted[(beta * k): min(beta * (k + 1), n)])

        w_curr = w_last - eta * partial_w(w_last, b_last, C, B)

        b_curr = b_last - eta * partial_b(w_last, b_last, C, B)
        k = int((k + 1) % num_batch)

        criterion_flag, delta_curr, loss_curr = MBGD_converge_criterion(delta_curr, w_last, w_curr, b_last, b_curr, C, eps, (X_permuted, y_permuted))
        MBGD_loss_history.append(loss_curr)

        w_last = w_curr
        b_last = b_curr

        if (t % 10 == 0):
            print(f"At iteration {t}, the loss is {loss_curr}")

    end = time.perf_counter()

    print(f"After {t} iterations, the MBGD algorithm converges with the final loss {loss_curr} and total runtime {end - start}.")
        
def main():
    BGD()
    SGD()
    MBGD()

if __name__ == "__main__":
    main()