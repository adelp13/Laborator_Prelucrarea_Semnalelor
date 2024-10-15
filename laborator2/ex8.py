import numpy as np
import matplotlib.pyplot as plt

aproximare_pade = lambda alpha: (alpha - (7 * (alpha**3) / 60)) / (1 + (alpha**2/20))

nr_esantioane = 1000
alpha = np.linspace(-np.pi/2, np.pi/2, nr_esantioane)
taylor = np.sin(alpha)
eroare_taylor = abs(taylor- alpha)
pade = aproximare_pade(alpha)
eroare_pade = abs(taylor - pade)

def creare_grafice(logaritmic): # ploteaza cele 4 subgrafice cu Oy normal si apoi Oy logaritmic
    fig, axs = plt.subplots(4, figsize=(8, 8))
    axs[0].set_title("sin(alpha) si alpha")
    axs[0].plot(alpha, taylor, label="sin(alpha)")
    axs[0].plot(alpha, alpha, label="alpha")
    axs[0].legend()
    axs[1].set_title("eroare = |sin(alpha) - alpha|")
    axs[1].plot(alpha, eroare_taylor)

    axs[2].set_title("sin(alpha) si pade")
    axs[2].plot(alpha, taylor, label="sin(alpha)")
    axs[2].plot(alpha, pade, label="pade")
    axs[2].legend()
    axs[3].set_title("eroare = |sin(alpha) - pade|")
    axs[3].plot(alpha, eroare_pade)

    if logaritmic:
        for ax in axs:
            ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

creare_grafice(False)
creare_grafice(True)