from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

def CalcularePredictii(serie_timp, p, nr_predictii, m):
    # y = Y * x
    # y[N - 1] il stim deja, scriem formula pt a il calulca
    N = len(serie_timp)
    ultimul_indice = N - 1
    predictii = np.zeros(nr_predictii)
    Y = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            Y[i][j] = serie_timp[ultimul_indice - 1 - i - j]

    y = np.zeros(m)
    for i in range(m):
        y[i] = serie_timp[ultimul_indice - i]
    # x_adj = inv(Y transp * Y) * Y transp * y
    x_adj = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(Y), Y)), np.transpose(Y)), y)
    x_adj_transp = np.transpose(x_adj)
    #print(len(x_adj))
    y = serie_timp[-p:]
    for i in range(nr_predictii):
        valoare_predictie = np.matmul(x_adj_transp, y)
        y = np.append(valoare_predictie, y[:-1])
        predictii[i] = valoare_predictie

    return predictii

N = 1000
a = 10
b = 0.1
c = 0.2
t = np.linspace(0, 1, N)
trend = np.zeros(N)
trend = a * t * t + b * t + c

sezon = 3 * np.sin(2 * np.pi * t * 10) + 2 * np.sin(2 * np.pi * t * 80)
variatii_mici = np.random.normal(0, 1.8, N)
serie_timp = trend + sezon + variatii_mici

