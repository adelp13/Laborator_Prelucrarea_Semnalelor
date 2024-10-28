import sounddevice as sd
import scipy.io.wavfile as wf
import numpy as np
import matplotlib.pyplot as plt

rate, semnal = wf.read('sounds/ex5.wav')

if semnal.ndim == 2:
    semnal = np.mean(semnal, axis=1)

# t = np.linspace(0, 10, 10 * rate) # am testat pt un sinus
# #print(t)
# f1 = 1000
# f2 = 2500
# f3 = 10000
# semnal = np.sin(2 * np.pi * f1 * t) + 3 * np.sin(2 * np.pi * f2 * t) + 4 * np.sin(2 * np.pi * f3 * t)

N = len(semnal)

N_grup = N // 100
suprapunere_grupuri = N_grup // 2
nr_grupuri  = (N - N_grup) // suprapunere_grupuri + 1
spectograma = np.zeros((nr_grupuri, N_grup // 2))

for i in range(0, nr_grupuri):
    grup = semnal[i * suprapunere_grupuri: i * suprapunere_grupuri + N_grup]
    spectograma[i] = np.abs(np.fft.fft(grup))[:N_grup // 2]

spectograma = (10 * np.log10(spectograma + 1e-8)).transpose()

plt.xlabel('timp (s)')
plt.ylabel('frecventa(kHz)')
img = plt.imshow(spectograma, aspect="auto", origin='lower', extent = [0, N // rate, 0, rate / 2 / 1000], cmap="plasma")
cbar = plt.colorbar(img)
cbar.set_label('prezenta frecventei (dB)')
plt.tight_layout()
plt.savefig("generated_images/ex6.pdf", format="pdf")
plt.show()