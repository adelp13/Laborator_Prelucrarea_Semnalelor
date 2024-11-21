from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

X = misc.face(gray=True)
Y = np.fft.fft2(X)
Y_shift = np.fft.fftshift(Y) # acum frecventele mici se afla in centru (cele pe care vrem sa le pastram)
# e mai usor sa le gasim asa decat daca se afla in cele 4 colturi

raza = 60 # cu cat e mai mare cu atat pastram mai multe frecvente, din ce in ce mai mari
filtru = np.zeros((len(Y), len(Y[0])))
for i in range(len(Y)):
    for j in range(len(Y[0])):
        if (i - ((len(Y) // 2)))**2 + (j - (len(Y[0]) // 2) )**2 <= raza ** 2:
            filtru[i][j] = 1

Y_shift_comprimat = Y_shift * filtru
Y_comprimat = np.fft.ifftshift(Y_shift_comprimat)
X_comprimat = np.fft.ifft2(Y_comprimat)
X_comprimat_real = np.real(X_comprimat)

plt.imshow(X_comprimat_real,cmap=plt.cm.gray)
plt.savefig('generated_images/ex2.pdf', format="pdf")
plt.show()

