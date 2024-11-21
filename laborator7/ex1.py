from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

N = 200
# 1 a
# x(n1, n2) = sin(2pi n1 + 3pi n2)
a_timp = np.zeros((N, N))
for i in range(a_timp[0].size):
    for j in range(a_timp[i].size):
        a_timp[i][j] = np.sin(2 * np.pi * i + 3 * np.pi * j)

# trecem in domeniul frecv:
a_frecventa = 10 * np.log10(np.real(np.abs(np.fft.fft2(a_timp))))
fig, axs = plt.subplots(2, 1, figsize=(9, 6))
axs[0].set_title("domeniul timp a)")
axs[0].imshow(a_timp, cmap=plt.cm.gray)
axs[1].set_title("domeniul frecventa a)")
axs[1].imshow(a_frecventa)
plt.savefig('generated_images/ex1_a.pdf', format="pdf")
plt.show()
# 1 b
# x(n1, n2) = sin(4pin1) + cos(6pi n2)
b_timp = np.zeros((N, N))
for i in range(b_timp[0].size):

    for j in range(b_timp[i].size):
        b_timp[i][j] = np.sin(4 * np.pi * i) + np.cos(6 * np.pi * j)
        #print(np.sin(4 * np.pi * i) + np.cos(6 * np.pi * j))

# trecem in domeniul frecv:
b_frecventa = 10 * np.log10(np.real(np.abs(np.fft.fft2(b_timp))) + 1e-8)
fig, axs = plt.subplots(2, 1, figsize=(9, 6))
axs[0].set_title("domeniul timp b)")
axs[0].imshow(b_timp, cmap=plt.cm.gray)
axs[1].set_title("domeniul frecventa b)")
axs[1].imshow(b_frecventa)
plt.savefig('generated_images/ex1_b.pdf', format="pdf")
plt.show()

# 1 c
#Y(0,5)=Y(0,N−5)=1, altfelYm1,m2=0,∀m1,m2

c_frecventa = np.zeros((N, N))
c_frecventa[0][5] = 1
c_frecventa[0][N - 5] = 1
# ne intoarcem in spatiul timp:
c_timp = np.fft.ifft2(c_frecventa)
c_timp = np.real(c_timp)

plt.imshow(c_timp, cmap=plt.cm.gray)
plt.title('1_c')
plt.show()
plt.savefig('generated_images/ex1_c.pdf', format="pdf")
# avem doar frecvente verticale, componenta lor orizontala e nula, de aceea imaginea in timp va avea un pattern vertical
# 1 d
#Y(5, 0)=Y(N−5, 0)=1, altfelYm1,m2=0,∀m1,m2

d_frecventa = np.zeros((N, N))
d_frecventa[5][0] = d_frecventa[N - 5][0] = 1

# ne intoarcem in spatiul timp:
d_timp = np.fft.ifft2(d_frecventa)
d_timp = np.real(d_timp)

plt.imshow(d_timp, cmap=plt.cm.gray)
plt.title('1_d')
plt.show()
plt.savefig('generated_images/ex1_d.pdf', format="pdf")
# 1 e
#Y(5, 5)=Y(N−5, N-5)=1, altfelYm1,m2=0,∀m1,m2
e_frecventa = np.zeros((N, N))
# e_frecventa[100][100] = 1
# e_frecventa[80][80] = 1
# e_frecventa[10][130] = 1
e_frecventa[N - 5][N - 5] = 1 # pe diagonala principala
e_frecventa[5][5] = 1
# ne intoarcem in spatiul timp:
e_timp = np.fft.ifft2(e_frecventa)
e_timp = np.real(e_timp)

plt.imshow(e_timp, cmap=plt.cm.gray)
plt.title('1_e')
plt.show()
plt.savefig('generated_images/ex1_e.pdf', format="pdf")
# frecventele au si componente verticale si orzontale
# pt ca 5 = 5 si N-5 = N-5 vor aparea doar linii diagonale,  altfel s-ar vedea si linii verticale si/sau orizontale
# cele 2 reprezinta aceeasi frecventa (scazand din N exact 5), deci daca pastram doar una rezulta aceeasi imagine