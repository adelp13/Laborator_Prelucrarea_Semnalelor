from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

X = misc.face(gray=True)
pixel_noise = 200

noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise

# eliminam zgomotul:
Y = np.fft.fft2(X_noisy)
Y_shift = np.fft.fftshift(Y)

raza = 35
filtru = np.zeros((len(Y), len(Y[0])))
for i in range(len(Y)):
    for j in range(len(Y[0])):
        if (i - ((len(Y) // 2)))**2 + (j - (len(Y[0]) // 2) )**2 <= raza ** 2:
            filtru[i][j] = 1

Y_shift_comprimat = Y_shift * filtru
Y_comprimat = np.fft.ifftshift(Y_shift_comprimat)
X_comprimat = np.fft.ifft2(Y_comprimat)
X_comprimat_real = np.real(X_comprimat)

fig, axs = plt.subplots(2, 1, figsize=(9, 6))
axs[0].set_title("noisy")
axs[0].imshow(X_noisy, cmap=plt.cm.gray)

axs[1].set_title("filtrat")
axs[1].imshow(X_comprimat_real, cmap=plt.cm.gray)
plt.savefig('generated_images/ex3.pdf', format="pdf")
plt.show()

SNR_inainte = 10 * np.log10(np.sum(X**2) / np.sum((noise**2)))
SNR_dupa = 10 * np.log10(np.sum(X**2) / np.sum((X_comprimat_real - X) **2))

print(f"snr inainte: {SNR_inainte}, snr dupa: {SNR_dupa}")
