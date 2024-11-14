import numpy as np
import matplotlib.pyplot as plt

N = 100

def convolutie(x, h):
    N = len(x)
    M = len(h)
    y = np.zeros(N + M - 1) # y[M - 1] e prima val care ia in calcul M taps si nu mai putin; y[N] e ultima; y[N + M - 1] va tine cont doar de x[N] * h[0]
    for i in range(N + M):
        for j in range(M):
            if 0 <= i - j < N: # verificam daca suntem in primele sau ultimele M valori unde adunam mai putin de M taps
                y[i] += x[i - j] * h[j]
    return y

x = np.random.rand(N)
# x2[n] = x[n] * x[0] + x[n - 1] * x[1] + x[n - 2] * x[2] + .... + x[0] * x[n]
# x2[n + 1] = x[n + 1] * x[0] + x[n] * x[1] + x[n - 1] * x[2] + .... + x[0] * x[n + 1]
# x2[a] inseamna semnalul inmultit cu el insusi decalat cu a (primele a taps inmultite cu ele dar invers), (a <= N),
# adica avem semnlaul 5 4 3 2 1 0 care se intalneste treptat tot cu el dar pozitionat 0 1 2 3 4 5; cand se suprapun complet taiem de la dreapta
# plotand se observa cum vectorul tinde spre o functie concava, pentru ca dimensiunea e 2 * N - 1 si in mijloc avem x[N] care are cele mai multe inmultiri, adica sanse mai mari pt o valoare mai mare
# deci cu cat mai multe convolutii cu atat se uniformizeaza mai mult plotarea, cresc sansele ca valorile din centru sa fie mai mari decat restul
# de asemenea inmultirile de pe margini sunt intre elementele de pe margini, adica elemente mai mici
# pe margini scade nr de inmultiri pana la 1
x2 = convolutie(x, x)
x3 = convolutie(x2, x)
x4 = convolutie(x3, x)
fig, axs = plt.subplots(4, 1, figsize=(10, 6))
axs[0].plot(x)
axs[0].set_title("vectorul initial")
axs[1].plot(x2)
axs[1].set_title("prima convolutie")
axs[2].plot(x3)
axs[2].set_title("a doua convolutie")
axs[3].plot(x4)
axs[3].set_title("a treia convolutie")
plt.tight_layout()
plt.savefig('generated_images/ex1.pdf', format="pdf")
plt.show()
