import numpy as np
import matplotlib.pyplot as plt

frecventa_semnale = 12
faze = [np.pi/36, np.pi/2, np.pi, (3 * np.pi) / 2]
semnale = []
t = np.linspace(0, 0.3, 1000)

plt.figure()
for i in range(len(faze)):
    semnale.append(np.sin(2 * np.pi * frecventa_semnale * t + faze[i]))
    plt.plot(t, semnale[i])

plt.show()

#adaugam zgomot la semnalul 1
SNR = [100, 10, 1, 0.1] # raportul dintre semnal si zgomot, daca e mare inseamna ca avem mult semnal

z = np.random.normal(0, 1, len(semnale[0])) # trebuie sa aiba aceeasi dimensiune ca semnalul la care aplica zgomot

fig, axs = plt.subplots(4)

# SNR = norm(x) ^ 2 / (gamma ^ 2) * (norm(z) ^ 2)
# gamma ^ 2 = norm(x) ^ 2 / SNR * (norm(z) ^ 2)
# gamma = norm(x) / sqrt(SNR) * norm(z)
# x = semnale[0]
# np.linalg.norm calculeaza in mod implicit norma de ordin 2

for i in range(len(SNR)):
    gamma = np.linalg.norm(semnale[0]) / (np.sqrt(SNR[i]) * np.linalg.norm(z))
    semnal1_zgomot = semnale[0] + gamma * z
    axs[i].plot(t, semnal1_zgomot)
    axs[i].set_title(f"SNR = {SNR[i]}")

plt.tight_layout()
plt.show()

