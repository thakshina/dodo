import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as nl

N = 3
Y0 = np.zeros((N, 4))  # startvektor som fås fra make_circle, Nx4 matrise
M = 4  # number of steps
K = np.zeros((N, 4))  # M stykker 4x4 matriser ?
h = 0.1  # stepsize


def sigma(Y):
    return np.tanh(Y)


def Euler(M, h, K, Y0):
    Ycurrent = Y0
    for i in range(M):
        Ynext = Ycurrent + h * sigma(Ycurrent * K)
        Ycurrent = Ynext
    return Ycurrent


Euler(M, h, K, Y0)


def etha(x):
    return np.exp(x) / (np.exp(x) + 1)


def J(W, C, M, h, K, Y0):
    YM = Euler(M, h, K, Y0)
    return 0.5 * nl.norm(etha(YM * W) - C, ord=2) ** 2  # YM*W og C er i R^4, YM er en Nx4 matrise og W en 4x1 vektor


