import numpy as np
import matplotlib.pyplot as plt


def CalculareMediereExponentiala(serie_timp, alpha):
    N = len(serie_timp)
    s = np.zeros(N)

    s[0] = serie_timp[0]
    for i in range(N):
        s[i] = (alpha * serie_timp[i]) + ((1 - alpha) * s[i - 1])
    return s

def CalculareAlphaPrinMinimizare(serie_timp):
    nr_alphas = 400
    alphas = np.linspace(0, 1, nr_alphas)

    suma_minima = 1000000000
    best_alpha = 0

    for i in range(nr_alphas):
        suma_de_minimizat = 0
        serie_noua = CalculareMediereExponentiala(serie_timp, alphas[i])
        for j in range(N - 1):
            suma_de_minimizat += (serie_noua[j] - serie_timp[j + 1]) ** 2
        if suma_de_minimizat < suma_minima:
            suma_minima = suma_de_minimizat
            best_alpha = alphas[i]
        #print(suma_de_minimizat)
    return best_alpha


# cream seria initiala de timp:
N = 1000
a = 0.8
b = 2.19
c = -0.5
t = np.linspace(0, 1, N)
trend = np.zeros(N)
trend = a * t * t + b * t + c

sezon = 3 * np.sin(2 * np.pi * t * 10) + 2 * np.sin(2 * np.pi * t * 80)
variatii_mici = np.random.normal(0, 1.9, N)
serie_timp = trend + sezon + variatii_mici

# aflam cle mai bun alpha prin minimizarea erorii:
best_alpha = CalculareAlphaPrinMinimizare(serie_timp)

serie_noua = CalculareMediereExponentiala(serie_timp, best_alpha)
plt.figure()
plt.title(f"Mediere exponentiala, best_alpha={best_alpha}")
plt.plot(t, serie_timp, label="seria initiala")
plt.plot(t, serie_noua, label="noua serie")
plt.xlabel("esantioane")
plt.ylabel("amplitudine")
plt.legend()
plt.savefig('generated_images/ex2.pdf', format="pdf")
plt.show()

