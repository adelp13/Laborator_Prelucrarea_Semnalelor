import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from scipy.datasets import ascent, face
from scipy.fft import dctn, idctn
from dahuffman import HuffmanCodec

dim_q = 8

vectorizare = [[], [], []] # pentru cele 3 canale Y Cb Cr

Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]]

def conversie_RGB_YCbCr(X):
    h, w, _ = X.shape
    X_YCbCr = np.zeros((h, w, 3))
    R = X[:, :, 0]
    G = X[:, :, 1]
    B = X[:, :, 2]

    X_YCbCr[:, :, 0] = 16 + (((65.738 * R) + (129.057 * G) + (25.064 * B)) / 256)
    X_YCbCr[:, :, 1] = 128 - (((37.945 * R) + (74.494 * G) - (112.439 * B)) / 256)
    X_YCbCr[:, :, 2] = 128 + (((112.439 * R) - (94.154 * G) - (18.285 * B)) / 256)

    return X_YCbCr

def conversie_YCbCr_RGB(X_YCbCr):
    h, w, _ = X_YCbCr.shape
    X = np.zeros((h, w, 3))

    Y = X_YCbCr[:, :, 0]
    Cb = X_YCbCr[:, :, 1] - 128
    Cr = X_YCbCr[:, :, 2] - 128

    X[:, :, 0] = Y + 1.402 * Cr
    X[:, :, 1] = Y - 0.344136 * Cb - 0.714136 * Cr
    X[:, :, 2] = Y + 1.772 * Cb

    X = np.clip(X, 0, 255)
    return X.astype(np.uint8)

def compresie_bloc(x):
    y = dctn(x)
    y_jpeg = Q_jpeg*np.round(y/Q_jpeg)
    return y_jpeg
def decompresie_bloc(y):
   # y = y_jpeg * Q_jpeg
    x = idctn(y)
    x = np.clip(x, 16, 240)
    return x
def vectorizare_bloc(y):
    vectorizare = []

    for i in range(2 * dim_q - 1):
        if i % 2 == 0:
            for j in range(min(i, dim_q - 1), max(-1, i - dim_q), - 1):
                vectorizare.append(y[j, i - j])
        else:
            for j in range(max(0, i - dim_q + 1), min(i + 1, dim_q)):
                vectorizare.append(y[j, i - j])
    return vectorizare

def devectorizare_bloc(vectorizare):
    y = np.zeros((dim_q, dim_q))
    index = 0

    for i in range(2 * dim_q - 1):
        if i % 2 == 0:
            for j in range(min(i, dim_q - 1), max(-1, i - dim_q), -1):
                y[j, i - j] = vectorizare[index]
                index += 1
        else:
            for j in range(max(0, i - dim_q + 1), min(i + 1, dim_q)):
                y[j, i - j] = vectorizare[index]
                index += 1
    return y

X = face()

# pasul 1: transformam din RGB in Y' Cb Cr
# Y' = Luma (intensitatea luminii, componenta grayscale)
# Chroma b = diferenta dintre canalul de albastru si Luma
# Chroma r = diferenta dintre rosu si  luma

X_YCbCr = conversie_RGB_YCbCr(X)

# pasii 2, 3, 4: pt fiecare dintre cele 3 canale aplicam dct + cuantizare pe blocuri 8 * 8, apoi zig zag + huffman :

h, w, c = X_YCbCr.shape

X_jpeg = np.zeros((h, w, c))

for k in range(0, c):
    for i in range(0, h - dim_q + 1, dim_q):
        for j in range(0, w - dim_q + 1, dim_q):
            x = X_YCbCr[i:i + dim_q, j:j + dim_q, k].copy()
            X_jpeg[i: i + dim_q, j: j + dim_q, k] = compresie_bloc(x)
            vectorizare[k].extend(vectorizare_bloc(X_jpeg[i: i + dim_q, j: j + dim_q, k]))

# codam cu huffman vectorul zig zag si salvam rezultatul intr-un fisier binar:
vectorizare_totala = np.concatenate(vectorizare)
len_vectorizare_canal = len(vectorizare[0])
codificator = HuffmanCodec.from_data(vectorizare_totala)
vectorizare_codata = codificator.encode(vectorizare_totala)

codificator.save("huffman.bin")

with open('fisier_binar.bin', 'wb') as f:
    f.write(vectorizare_codata)


# acum recream imaginea initiala facand procesul invers:

with open('fisier_binar.bin', 'rb') as f:
    vectorizare_codata = f.read()

codificator = HuffmanCodec.load("huffman.bin")
vectorizare_totala_decodata = codificator.decode(vectorizare_codata)

vectorizare_decodata = [[], [], []]
vectorizare_decodata[0] = vectorizare_totala_decodata[:len_vectorizare_canal]
vectorizare_decodata[1] = vectorizare_totala_decodata[len_vectorizare_canal: 2 * len_vectorizare_canal]
vectorizare_decodata[2] = vectorizare_totala_decodata[2 * len_vectorizare_canal:]

# acum vrem sa obtinem imaginea initiala in cod YBcCr (pt fiecare bloc facem opusul vectorizarii, cuantizarii, iar apoi idctn

X_decodat_YCbCr = np.zeros((h, w, c))

for k in range(0, c):
    index_vectorizare = 0
    for i in range(0, h - dim_q + 1, dim_q):
        for j in range(0, w - dim_q + 1, dim_q):
            X_decodat_YCbCr[i:i + dim_q, j:j + dim_q, k] = devectorizare_bloc(vectorizare_decodata[k][index_vectorizare:index_vectorizare + dim_q * dim_q])
            X_decodat_YCbCr[i:i + dim_q, j:j + dim_q, k] = decompresie_bloc(X_decodat_YCbCr[i:i + dim_q, j:j + dim_q, k])
            index_vectorizare += dim_q * dim_q

# diferenta = np.abs(X_YCbCr - X_decodat_YCbCr) # pt testat devectorizarea
# trecem inapoi in RGB:
X_decodat = conversie_YCbCr_RGB(X_decodat_YCbCr)
#X_decodat = conversie_YCbCr_RGB(X_YCbCr)

plt.imshow(X, cmap=plt.cm.gray)
plt.title("imagine initiala")
plt.savefig('generated_images/ex2_imagine_initiala.png', format="png")
plt.show()
plt.imshow(X_decodat)
plt.title("imagine decodata")
plt.savefig('generated_images/ex2_imagine_decodata.png', format="png")
plt.show()


