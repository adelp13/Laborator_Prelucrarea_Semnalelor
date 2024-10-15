import numpy as np
import matplotlib.pyplot as plt

frecventa_esantionare = 1500
frecventa1 = frecventa_esantionare / 2
frecventa2 = frecventa_esantionare / 4
frecventa3 = 0

durata_semnale = 1
t = np.linspace(0, durata_semnale, int(durata_semnale * frecventa_esantionare))
sinus1 = np.sin(2 * np.pi * frecventa1 * t)
sinus2 = np.sin(2 * np.pi * frecventa2 * t)
sinus3 = np.sin(2 * np.pi * frecventa3 * t)

#Observatii:
# toate cele 3 semnale au frecventa de la jumatatea frecventei de esantionare in jos, deci pot fi reconstruite
# un semnal cu f < fs/2 nu garanteaza un grafic interpolat perfect, dar e suficient pentru a retine informatiile esentiale pt sin
# (frecv = n => 2 * n schimbari de directii in graful sinusoidal, de aceea regula fs >= 2 * f)

# pentru primul semnal (fs/2) avem la limita destule esantioane, dar reprezentarea este destul de departe de realitate
# la semnalul fs/4 se vede mai clar, dar nu pare sa fie suficient nici acolo
# semnalul 3 are toate esantioanele cu y = A * faza = 0 pentru ca frecventa nula anuleaza tot mai putin faza
# aici putem folosi si un singur esantion
# orice semnal cu frecventa nula este o functie continua

fig, axs = plt.subplots(3)
axs[0].plot(t, sinus1)
#axs[0].stem(t, sinus1)
axs[0].set_title("f/2")
axs[1].plot(t, sinus2)
#axs[1].stem(t, sinus2)
axs[1].set_title("f/4")
axs[2].plot(t, sinus3)
#axs[2].stem(t, sinus3)
axs[2].set_title("0")
plt.tight_layout()
plt.show()




