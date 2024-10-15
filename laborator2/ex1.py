import numpy as np
import matplotlib.pyplot as plt


t = np.linspace(0, 1, 1000)
A = 3
faza = np.pi/3
frecventa_semnal = 12
semnal_sin = A * np.sin(2 * np.pi * frecventa_semnal * t + faza + np.pi/2)
# faza celui cu sin e mai mare cu np.pi/2 ca sa se suprapuna cu cos
semnal_cos = A * np.cos(2 * np.pi * frecventa_semnal * t + faza)

fig, axs = plt.subplots(2)
axs[0].plot(t, semnal_sin)
axs[0].set_title("sinusoidal")
axs[1].plot(t, semnal_cos)
axs[1].set_title("cosinusoidal")

plt.tight_layout()
plt.show()
