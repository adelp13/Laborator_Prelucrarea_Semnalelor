import numpy as np
import matplotlib.pyplot as plt
import math

N = 8

F = np.zeros((N, N), dtype="complex")

for l in range(N):
    for p in range(N):
        F[l][p] = math.e**(-2 * np.pi * 1j * l * p / N)

fig, axs = plt.subplots(N, figsize=(8, 8))

for l in range(N):
    axs[l].plot(F[l].real, label="real")
    axs[l].plot(F[l].imag, label="imaginar")
    axs[l].legend()

plt.tight_layout()
plt.savefig("generated_images/ex1.pdf", format="pdf")
plt.show()

# F * FH = N * I(N)

# F = 1/sqrt(n) * M;    M * MH = I(N)
# M * MH = F * FH / N
FH = np.transpose(np.conjugate(F))
matricea_identitate = np.eye(N)
produs_F_FH = np.dot(FH, F) / N

print(np.allclose(produs_F_FH, matricea_identitate))












