import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 1000)
frecventa_semnale = 10 # le am pus aceeasi frecventa ca sa fie uniform semnalul rezultat
sinus = np.sin(2 * np.pi * frecventa_semnale * t + np.pi/5)
sawtooth = t * frecventa_semnale - np.floor(t * frecventa_semnale)
suma_semnale = sinus + sawtooth

fig, axs = plt.subplots(3)
axs[0].plot(t, sinus)
axs[1].plot(t, sawtooth)
axs[2].plot(t, suma_semnale)
plt.tight_layout()
plt.show()

