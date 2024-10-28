import numpy as np
import matplotlib.pyplot as plt
import math


frecventa_semnal = 20
N = 1000
fs = 3 # sub 2 * 20
durata_semnale = 1

t = np.linspace(0, durata_semnale, N)
esantionare = np.linspace(0, durata_semnale, int(durata_semnale * fs) + 1)

fs -= 1
semnal_initial = np.sin(2 * np.pi * frecventa_semnal * t)
semnal_initial_esantionat = np.sin(2 * np.pi * frecventa_semnal * esantionare)

semnal_a = np.sin(2 * np.pi * (frecventa_semnal - 3 * fs) * t)
semnal_a_esantionat = np.sin(2 * np.pi * (frecventa_semnal - 3 * fs) * esantionare)

semnal_b = np.sin(2 * np.pi * (frecventa_semnal - 6 * fs) * t)
semnal_b_esantionat = np.sin(2 * np.pi * (frecventa_semnal - 6 * fs) * esantionare)

fig, axs = plt.subplots(4, figsize=(4, 6))
axs[0].plot(t, semnal_initial)

axs[1].plot(t, semnal_initial)
axs[1].stem(esantionare, semnal_initial_esantionat)

axs[2].plot(t, semnal_a)
axs[2].stem(esantionare, semnal_a_esantionat)

axs[3].plot(t, semnal_b)
axs[3].stem(esantionare, semnal_b_esantionat)

plt.tight_layout()
plt.savefig("generated_images/ex2.pdf", format="pdf")
plt.show()

# se vede in subgraficul 2 cum esantioanele nu sunt suficiente sa mentina forma semnalului initial
# pentru semnale a caror relatie intre frecvente este f1 +/- k*fs = f2, esantioanele vor fi identice pt ca sin e functie periodica, deci va reiesi acelasi semnal discret
