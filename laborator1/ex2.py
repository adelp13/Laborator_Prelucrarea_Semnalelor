import numpy as np
import matplotlib.pyplot as plt

def functie_f(i, j):
    return i + j

#a: semnal sinusoidal de frecventa 400Hz, care sa contina 1600 esantioane

frecventa_semnal = 400
nr_esantioane = 1600
t = np.linspace(0, 1, nr_esantioane)
a = np.sin(2 * np.pi * t * frecventa_semnal + np.pi/3)
plt.plot(t, a)
plt.stem(t, a)
plt.title('a)')
plt.show()

#b: Un semnal sinusoidal de frecventa 800 Hz, care sa dureze 3 secunde

frecventa_semnal = 800
durata_semnal = 3
t = np.linspace(0, durata_semnal, 150)

b = np.sin(2 * np.pi * frecventa_semnal * t + np.pi/16)

plt.plot(t, b)
plt.stem(t, b)
plt.title('b)')
plt.show()


#c: Un semnal de tip sawtooth de frecventa 240 Hz (puteti folosi functiile numpy.floor sau numpy.mod)
# trebuie o functie care urca si apoi se intoarce la valoarea initiala
frecventa_semnal = 240
t = np.linspace(0, 0.1, 100000)

# forma initiala: c = t - np.floor(t)
# cand timpul e < 1 valoarea va creste liniar pentru ca np.floor returneaza 0
# momentan avem o urcare in fiecare secunda, vrem 240 per secunda

c = t * frecventa_semnal - np.floor(t * frecventa_semnal)
# acum
plt.plot(t, c)
plt.title('c)')
plt.show()


#d: Un semnal de tip square de frecventa 300 Hz (puteti folosi functia numpy.sign)

t = np.linspace(0, 0.1, 1000)
frecventa_semnal = 300
d = np.sign(np.cos(2 * np.pi * frecventa_semnal * t))

plt.plot(t, d)
plt.title('d)')
plt.show()

#e: semnal 2D aleator
e = np.random.rand(128, 128)
plt.imshow(e)
plt.title('e)')
plt.show()

#f: semnal 2D la alegere
f = np.zeros((128, 128))

for i in range(f[0].size):
    for j in range(f[i].size):
        f[i][j] = functie_f(i, j)
plt.imshow(f)
plt.title('f)')
plt.show()