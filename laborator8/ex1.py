from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt


def CalculareAutocorelatie(serie_timp):
    N = len(serie_timp)
    autocorelatie = np.zeros(2 * N - 1)

    for lag in range(-N + 1, N):
        autocorelatie[lag + N - 1] = np.sum(serie_timp[:N - abs(lag)] * serie_timp[abs(lag):])
    # se decaleaza cu abs(lag)
    return autocorelatie

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
# a
a = 10
b = 0.1
c = 0.2
t = np.linspace(0, 1, N)
trend = np.zeros(N)
trend = a * t * t + b * t + c

sezon = 3 * np.sin(2 * np.pi * t * 10) + 2 * np.sin(2 * np.pi * t * 80)
variatii_mici = np.random.normal(0, 0.8, N)
serie_timp = trend + sezon + variatii_mici
fig, axs = plt.subplots(4, 1, figsize=(9, 6))
axs[0].set_title("trend")
axs[0].plot(t, trend)
axs[0].set_xlabel("esantioane")
axs[0].set_ylabel("amplitudine")
axs[1].set_title("sezon")
axs[1].plot(t, sezon)
axs[1].set_xlabel("esantioane")
axs[1].set_ylabel("amplitudine")
axs[2].set_title("variatii mici")
axs[2].plot(t, variatii_mici)
axs[2].set_xlabel("esantioane")
axs[2].set_ylabel("amplitudine")
axs[3].set_title("serie timp")
axs[3].plot(t, serie_timp)
axs[3].set_xlabel("esantioane")
axs[3].set_ylabel("amplitudine")
plt.tight_layout()
plt.savefig('generated_images/ex1_a.pdf', format="pdf")
plt.show()

# b
autocorelatie_numpy = np.correlate(serie_timp, serie_timp, 'full')
autocorelatie = CalculareAutocorelatie(serie_timp)
lags = np.linspace(-N + 1, N - 1, 2 * N - 1)
plt.figure()
plt.plot(lags, autocorelatie_numpy, label="functia numpy")
plt.plot(lags, autocorelatie, label="functe definita")
plt.xlabel("laguri")
plt.ylabel("corelatie")
plt.legend()
plt.savefig('generated_images/ex1_b.pdf', format="pdf")
plt.show()

# c
p = 100
nr_predictii = 200
predictii = CalcularePredictii(serie_timp, p, nr_predictii, N)
serie_cu_predictii = np.concatenate((serie_timp, predictii))
t_cu_predictii = np.linspace(0, 1 + (nr_predictii / N), N + nr_predictii)
frecv_esantionare = 1 / N
plt.figure()

plt.plot(t_cu_predictii, serie_cu_predictii, label="predictii")
plt.plot(t, serie_timp, label="semnal intial")
plt.xlabel("esantioane")
plt.ylabel("amplitudine")
plt.legend()
# print(predictii)
plt.savefig('generated_images/ex1_c.pdf', format="pdf")
plt.show()

# d
ps = [1, 2, 4, 5, 10, 30]
ms = [10, 20, 30, 40]

n = 100
MSEs = np.zeros((len(ps), len(ms))) # calculam MSE pentru fiecare pereche de p si m, luand ultimele 100 esantioane cunoscute din seria de timp
# mean squared error
best_MSE = 100000000000
best_p, best_p = 0, 0

for index_p in range(len(ps)):
    for index_m in range(len(ms)):
        for i in range(1, n + 1):
            serie_timp_decupata = serie_timp[:-i] # adica urmatorul element ar fi serie_timp[N - i]
            predictie = CalcularePredictii(serie_timp_decupata, ps[index_p], 1, ms[index_m])
            pred = predictie[0] # trebuie sa o comparam cu serie_timp[N - i], care e valoarea adevarata
            MSEs[index_p][index_m] += (1 / n) * ((serie_timp[N - i] - pred)**2)
        #print(MSEs[index_p][index_m])

        if (MSEs[index_p][index_m] < best_MSE):
            best_MSE = MSEs[index_p][index_m]
            best_m, best_p = ms[index_m], ps[index_p]

print(f"valoarile optime pentru m si p sunt {best_m} si {best_p}, obtinand un MSE de {best_MSE}")

