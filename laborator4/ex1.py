import numpy as np
import matplotlib.pyplot as plt
import math
import time

dimensiuni = [2**i for i in range(7, 13)]
N = 1000
f1 = 2
f2 = 10
f3 = 20 # < 128/2

# daca am N esantioane, pot calcula X pentru N frecvente diferite
# f[0] = 0 Hz, f[1] = fs * 1 / N...
# deci nu putem gasi frecvente > fs, dar oricum ar fi aparut aliasing la f > f2 / 2

def calculateTF(N, semnal):
    X = np.zeros((N), dtype="complex")
    for frecv in range(N):
        for timp in range(N):
            X[frecv] += (semnal[timp] * math.e ** (-2 * np.pi * 1j * frecv * timp / N))

timpi_FT_laborator = [0] * len(dimensiuni)
timpi_FFT = [0] * len(dimensiuni)

for i in range(len(dimensiuni)):
    t = np.linspace(0, 1, dimensiuni[i])
    semnal = np.sin(2 * np.pi * f1 * t) + 3 * np.sin(2 * np.pi * f2 * t) + 4 * np.sin(2 * np.pi * f3 * t)

    start = time.perf_counter()
    calculateTF(dimensiuni[i], semnal)
    end = time.perf_counter()
    timpi_FT_laborator[i] = (end - start)

    start = time.perf_counter()
    np.fft.fft(semnal)
    end = time.perf_counter()
    timpi_FFT[i] = (end - start)
    print(dimensiuni[i], timpi_FFT[i], timpi_FT_laborator[i])

plt.figure()
plt.plot(dimensiuni, timpi_FT_laborator, label="FT laborator")
plt.plot(dimensiuni, timpi_FFT, label="np.fft.fft()")
plt.xlabel('esantioane')
plt.ylabel('timp executie')
plt.yscale('log')
plt.legend()
plt.savefig('generated_images/ex1.pdf', format='pdf')
plt.show()









