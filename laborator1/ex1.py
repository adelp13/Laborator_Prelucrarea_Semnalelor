import numpy as np
import matplotlib.pyplot as plt


#a
axa_timp = np.arange(0, 0.03, 0.0005)

#b
x_continuu = np.cos(520 * np.pi * axa_timp + np.pi/3) # np.pi/3 este faza
y_continuu = np.cos(280 * np.pi * axa_timp - np.pi/3)
z_continuu = np.cos(120 * np.pi * axa_timp + np.pi/3)

fig, axs = plt.subplots(3)
# avem 2 tipuri diferite de frecvente, a semnalului si cea de esantionare

fig.suptitle('Semnale continue')

axs[0].plot(axa_timp, x_continuu)
axs[0].set_title('x(t)')
axs[1].plot(axa_timp, y_continuu)
axs[1].set_title('y(t)')
axs[2].plot(axa_timp, z_continuu)
axs[2].set_title('z(t)')
plt.tight_layout()
plt.show()

#c

frecventa_esantionare = 200
perioada_esantionare = 1 / frecventa_esantionare
esantionare = np.arange(0, 0.03, perioada_esantionare)
# arange are ca ultim parametru pasul, in loc de numarul de elemente cum are linspace
x_esantionat = np.cos(520 * np.pi * esantionare + np.pi/3) # pe grafic nu se observa diferente intre cele 3 semnale decat la cresterea frecventei de esantionare, 200 e prea putin
y_esantionat = np.cos(280 * np.pi * esantionare - np.pi/3)
z_esantionat = np.cos(120 * np.pi * esantionare + np.pi/3)

fig, axs = plt.subplots(3)
fig.suptitle('Semnale eșantionate la 200 Hz')

axs[0].stem(esantionare, x_esantionat) # stem pentru esantionare
axs[0].plot(esantionare, x_esantionat)
axs[0].set_title('x[n]')
axs[1].stem(esantionare, y_esantionat)
axs[1].plot(esantionare, y_esantionat)
axs[1].set_title('y[n]')
axs[2].stem(esantionare, z_esantionat)
axs[2].plot(esantionare, z_esantionat)
axs[2].set_title('z[n]')
plt.tight_layout()
plt.show()
