import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from scipy.datasets import ascent
import pooch

from scipy.fft import dctn, idctn

dim_q = 8

def compresie_bloc(x):
    Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 28, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]]

    y = dctn(x)
    y_jpeg = Q_jpeg*np.round(y/Q_jpeg)
    x_jpeg = idctn(y_jpeg)

    return x_jpeg

X = ascent()

Q_down = 10
X_jpeg = X.copy()
X_jpeg = Q_down*np.round(X_jpeg/Q_down)

h, w = X_jpeg.shape

for i in range(0, h - dim_q + 1, dim_q):
    for j in range(0, w - dim_q + 1, dim_q):
        x = X_jpeg[i:i + dim_q, j:j + dim_q].copy()
        X_jpeg[i: i + dim_q, j: j + dim_q] = compresie_bloc(x)

plt.imshow(X, cmap=plt.cm.gray)
plt.savefig('generated_images/ex1_imagine_initiala.pdf', format="pdf")
plt.show()
plt.imshow(X_jpeg, cmap=plt.cm.gray)
plt.savefig('generated_images/ex1_jpeg.pdf', format="pdf")
plt.show()
