import math

import numpy as np
import matplotlib.pyplot as plt

N = 10000
# cu cat esantionez mai mult cu atat frecvetele care nu se afla in semnal vor avea modului mai apropiat de 0
t = np.linspace(0, 1, N)
#print(t)
f1 = 10
f2 = 25
f3 = 100
omega_maxim = 120
semnal = np.sin(2 * np.pi * f1 * t) + 3 * np.sin(2 * np.pi * f2 * t) + 4 * np.sin(2 * np.pi * f3 * t)

X = np.zeros((omega_maxim + 1), dtype="complex")
for frecv in range(omega_maxim + 1):
    for timp in range(N):
        X[frecv] += (semnal[timp] * math.e**(-2 * np.pi * 1j * frecv * timp / N))

abs_X = np.abs(X)
omegas = np.linspace(0, omega_maxim, omega_maxim + 1)

fig, axs = plt.subplots(2, figsize=(4, 6))
axs[0].axhline()
axs[0].plot(t, semnal)
axs[0].set_xlabel('Timp')
axs[0].set_ylabel('x(t)')

axs[1].stem(omegas, abs_X)
axs[1].set_xlabel('|X(w)|')
axs[1].set_ylabel('Frecventa')

plt.tight_layout()
plt.savefig("generated_images/ex3.pdf", format="pdf")
plt.show()


