import numpy as np
import matplotlib.pyplot as plt
import math


frecventa_semnal = 5
N = 1000
fs = 15 # peste 2 * 5
durata_semnale = 1

t = np.linspace(0, durata_semnale, N)
esantionare = np.linspace(0, durata_semnale, int(durata_semnale * fs) + 1)

fs -= 1
semnal_initial = np.sin(2 * np.pi * frecventa_semnal * t)
semnal_initial_esantionat = np.sin(2 * np.pi * frecventa_semnal * esantionare)

semnal_a = np.sin(2 * np.pi * (frecventa_semnal + fs) * t)
semnal_a_esantionat = np.sin(2 * np.pi * (frecventa_semnal + fs) * esantionare)

semnal_b = np.sin(2 * np.pi * (frecventa_semnal + 2 * fs) * t)
semnal_b_esantionat = np.sin(2 * np.pi * (frecventa_semnal + 2 * fs) * esantionare)

fig, axs = plt.subplots(4, figsize=(4, 6))
axs[0].plot(t, semnal_initial)

axs[1].plot(t, semnal_initial)
axs[1].scatter(esantionare, semnal_initial_esantionat)

axs[2].plot(t, semnal_a)
axs[2].scatter(esantionare, semnal_a_esantionat)

axs[3].plot(t, semnal_b)
axs[3].scatter(esantionare, semnal_b_esantionat)

plt.tight_layout()
plt.savefig("generated_images/ex3.pdf", format="pdf")
plt.show()

# semnalele au aceleasi esatioane, dar primul poate fi distins de restul pentru ca fs > 2 * f
# adaugand fs si 2 fs la semnalele a si b, niciunul nu mai poate avea f <= fs/2; au aceeasi esantionare ca primul dar pentru ele esantionarea este prea mica
