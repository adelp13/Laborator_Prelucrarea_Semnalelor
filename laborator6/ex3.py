import numpy as np
import matplotlib.pyplot as plt

N = 1000
t = np.linspace(0, 1, N)
sinus = np.sin(2 * np.pi * 100 * t)
Nw = 200

def fereastra_hanning(Nw):
    fereastra = np.zeros(Nw)
    for i in range(Nw):
        fereastra[i] = 0.5 * (1 - np.cos(2 * np.pi * i / Nw)) # se concetreaza mai mult pe frecventele din centru, cele de pe margini tind spre 0, cele din mijloc au pondere spre 1
    return fereastra

aplicare_dreptunghi = np.concatenate([np.ones(Nw), np.zeros(N - Nw)])
aplicare_hanning = np.concatenate([np.zeros(N - Nw), fereastra_hanning(Nw)])
sinus_dreptunghi = aplicare_dreptunghi * sinus
sinus_hanning = aplicare_hanning * sinus

fig, axs = plt.subplots(3, 1, figsize=(10, 6))
axs[0].plot(t, sinus)
axs[0].set_title("semnalul initial")
axs[1].plot(t, sinus_dreptunghi)
axs[1].set_title("fereastra dreptunghiulara")
axs[2].plot(t, sinus_hanning)
axs[2].set_title("fereastra Hanning")
plt.tight_layout()
plt.savefig('generated_images/ex3.pdf', format="pdf")
plt.show()