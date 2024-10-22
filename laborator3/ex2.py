import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 8000)
frecventa_semnal = 5
esantion = 6200

sinus = np.sin(2 * np.pi * frecventa_semnal * t)
distanta_origine0 = np.sqrt(t**2 + sinus**2) # pentru a afisa cu o culoare diferita
y_n = sinus * np.exp(-2 * np.pi * 1j * t)
distanta_origine1 = np.sqrt(y_n.real ** 2 + y_n.imag ** 2)

fig, axs = plt.subplots(2, figsize=(4, 6))
axs[0].axhline()
axs[0].scatter(t, sinus, c=distanta_origine0, s=1)
axs[0].set_xlabel('Timp')
axs[0].set_ylabel('Amplitudine')
axs[0].scatter(t[esantion], sinus[esantion], color='red')

axs[1].axhline()
axs[1].axvline()
axs[1].scatter(y_n.real, y_n.imag, c=distanta_origine1, s=1)
axs[1].set_xlabel('Real')
axs[1].set_ylabel('Imaginar')
axs[1].set_ylim(-1, 1)
axs[1].set_xlim(-1, 1)
axs[1].scatter(y_n.real[esantion], y_n.imag[esantion], color='red')

plt.tight_layout()
plt.savefig("generated_images/ex2_figura1.pdf", format="pdf")
plt.show()

frecv_infasurare = [2, 3, 5, 8]
fig, axs = plt.subplots(2, 2, figsize=(9, 11))
fig.suptitle(f"frecventa semnal={frecventa_semnal}")

for i in range(len(frecv_infasurare)):
    linie = (i // 2)
    coloana = i % 2

    y_n = sinus * np.exp(-2 * np.pi * 1j * t * frecv_infasurare[i])
    distanta_origine1 = np.sqrt(y_n.real ** 2 + y_n.imag ** 2)
    axs[linie, coloana].set_title(f"omega={frecv_infasurare[i]}")
    axs[linie, coloana].axhline()
    axs[linie, coloana].axvline()
    axs[linie, coloana].scatter(y_n.real, y_n.imag, c=distanta_origine1, s=1)
    axs[linie, coloana].set_xlabel('Real')
    axs[linie, coloana].set_ylabel('Imaginar')
    axs[linie, coloana].scatter(y_n.real[esantion], y_n.imag[esantion], color='red')
    axs[linie, coloana].set_ylim(-1, 1)
    axs[linie, coloana].set_xlim(-1, 1)

plt.tight_layout()
plt.savefig("generated_images/ex2_figura2.pdf", format="pdf")
plt.show()