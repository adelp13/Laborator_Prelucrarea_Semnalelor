import numpy as np
import matplotlib.pyplot as plt
import math
import csv

N = 18288 # esantioane
# a
# frecventa de esantionare este de 1/3600 Hz (dupa fiecare ora se numara cate masini sunt in acel moment in intersectie)
fs = 1/3600 # pentru a o avea in Hz
# b
ore = 18288 # pentru ca avem rata de un esantion la ora
zile = ore / 24
# esantiaonele acopera 762 zile

# c
# frecventa maxima este B = fs/2 = 1/3600/2 = 1/7200 Hz
# o frecventa mai mare nu poate fi reprezentata clar. Nu exista destule esantioane pentru a retine forma semnalului
frecv_max = 0.5 * fs

# d
x = np.genfromtxt('kaggle/Train.csv', delimiter=',', skip_header=1, usecols=-1, dtype=int)

X = np.fft.fft(x)
X = abs(X/N)
X = X[:N//2]

# vrem sa plotam si avem nevoie de frecventele cautate de fft:
f = fs * np.linspace(0, N//2, N//2) / N # imparitm pt a normaliza in raport cu nr de esantioane

plt.figure()
plt.yscale('log')
plt.plot(f, X)
plt.title('ex1_d')
plt.savefig("generated_images/ex1_d.pdf", format="pdf")
plt.show()
#print(X)

# e
# Exista o componenta continua pentru ca pe graficul fft apare o valoare mare pentru frecventa 0. Componenta continua nu are frecventa, e un semnal constant.
X = np.fft.fft(x - np.mean(x))
X = abs(X/N)
X = X[:N//2]

fig, axs = plt.subplots(2, 1, figsize=(10, 6))
axs[0].plot(f, X)
axs[0].set_title('fft')
axs[1].plot(np.linspace(0, N, N), x - np.mean(x))
axs[1].set_title('semnal')
plt.tight_layout()
plt.savefig("generated_images/ex1_e.pdf", format="pdf")
plt.show()
# f
indici_sortati = sorted(np.argsort(X)[-4:])

for i in indici_sortati:
    print(f"frecventa {f[i]} Hz cu modului transformatei = {X[i]}, cu perioada {1 / f[i]} secunde, adica {1 / (f[i] * 3600)} ore si {1 / (f[i] * 3600 * 24)} zile.")
# rezultat:
# frecventa 1.5190734866989925e-08 Hz cu modului transformatei = 66.85385766393449, cu perioada 65829600.0 secunde, adica 18286.0 ore si 761.9166666666667 zile.
# frecventa 3.038146973397985e-08 Hz cu modului transformatei = 35.2191729779369, cu perioada 32914800.0 secunde, adica 9143.0 ore si 380.95833333333337 zile.
# frecventa 4.557220460096978e-08 Hz cu modului transformatei = 25.21991648404482, cu perioada 21943199.999999996 secunde, adica 6095.333333333333 ore si 253.97222222222217 zile.
# frecventa 1.1575339968646325e-05 Hz cu modului transformatei = 27.10202228761556, cu perioada 86390.55118110235 secunde, adica 23.997375328083987 ore si 0.9998906386701661 zile.

# Interpretare:
# Prima frecventa: o componenta de semnal de perioada 761 zile sugereaza un pattern care se repeta la aproximativ 2 ani. Aceasta este perioada completa pe care s-a facut studiul
# A doua frecventa: perioada putin mai mare de un an ar putea reprezenta un an, adica schimbari in functie de anotimpuri
# A treia: aproximativ un an jumate
# Ultima: o zi (perioada de 0.9998 zile)
# Aceste frecvente sunt cele mai importante pentru forma semnalului, avand magnitudinile cele mai mari
# Cea de frecventa 0 (componenta continua) nu mai apare pentru ca am eliminat-o

# g
# o luna de trafic inseamna 24 * 28 = 672 ore adica 672 esantioane
nr_esantioane_luna = 672
# din fisierul Train.csv observam linia 2232,26-11-2012 00:00, 8
# adica prima inregistrare din ziua de luni 26 11 2012, esantionul 2233 cu indicele 2232

indice_start = 2232
indice_final = indice_start + nr_esantioane_luna - 1
x_luna = x[indice_start:indice_final + 1]
esantioane = np.linspace(indice_start, indice_final, nr_esantioane_luna)

plt.figure()
plt.plot(esantioane, x_luna)
plt.title('ex1_g')
plt.savefig("generated_images/ex1_g.pdf", format="pdf")
plt.show()

# h
# pentru a aproxima data inceperii studiului, putem cauta diverse comportamente periodice ca la punctul f
# afisam separat aceste frecvente pe axa timpului si ne uitam daca prima perioada este completa
# frecventele de perioada zilnica pot sugera ora la care a inceput graficul, esantioanele fiind din ora in ora. Deci o perioada zilnica va avea 24.
# daca prima perioada este completa, inseamna ca se incepe de la ora 00. putem aproxima ora si in functie de amplitudinea componentei de frecventa la punctul de inceput, care este mai mare la orele de varf
# perioada de un an poate determina anotimpul. De exemplu iarna ar putea fi mai putin trafic, adica la inceputul si finalul perioadei.
# tot cu perioada de un an putem determina direct data cu luna si zi calculand al catelea esantion din perioada este punctul de start
# pot exista frecvente care au o magnitutdine mult mai mare in anumite zile, cum ar fi anumite sarbatori sau evenimente. FFT poate determina prezenta diverselor componente de frecventa de-a lungul studiului.

# nu putem afla mereu data exacta, mai ales cand studiul nu se intinde pe o perioada prea mare de timp.
# numarand esantioanele putem afla durata studiului. daca avem mai putin de un an este greu sa aflam luna sau ziua pentru ca fft nu va gasi perioada anuala. Daca avem mai putin de o zi nu putem afla nici ora.
# daca avem jumatate de an putem aproxima luna in functie de cat trafic avem, dar nu va fi un rezultat exact si avem nevoie sa comparam cu alte esantioane din studiu (de exemplu observam ca ar putea fi iarna pentru ca dupa creste traficul).
# cel mai greu de determinat este anul in care a inceput, pentru ca perioadele mai mari de un an sunt rare.
# chiar si cu esantioane care se intind pe mai multi ani, este greu de determinat anu exact.
# putem sa analizam schimbari in infrastructura (de ex plasarea mai multor semafoare) si data cand au avut loc si sa o corelam cu valori mai mari sau mai mici ale traficului dupa acea data si vizibil mai mici inainte, si sa calculam cate esantioae ar fi intre esantioul de start si cel unde incepe sa se produca
# dar nici aceasta metoda nu este precisa, schimbari de infrastructura au loc destul de des.

# i

# frecventele inalte pot sugera schimbari prea bruste care nu ajuta la analiza traficului (frecvente mai mici de o zi).
# frecventele le consideram in secunde
# eliminam toate frecventele mai mari de o valoare; perioada minima va fi de 23 de ore = 82800 secunde, adica o frecventa mai mica de 1/82800 Hz
frecventa_maxima = 1/82800

# pentru a elimina aceste frecvente trebuie sa eliminam din vectorul fft si apoi sa nu le includem in fft invers
X = np.fft.fft(x)
X = abs(X/N) # nu mai eliminam a doua jumatate, ne va trebui pt a ne intoarce la semnalul filtrat (cu ifft)
f = fs * np.linspace(0, N, N) / N
X_filtrare = np.where(f <= frecventa_maxima, X, 0 + 0j)
x_filtrare = np.fft.ifft(X_filtrare).real # .real pt ca intorcandu-ne din fft functia nu stie daca semnalul avea de la inceput componente complexe sau nu
esantioane = np.linspace(0, N, N)

indice1 = 100
indice2 = 1000

fig, axs = plt.subplots(2, 1, figsize=(10, 6))
axs[0].plot(esantioane[indice1:indice2], x_filtrare[indice1:indice2])
axs[0].set_title('semnal filtrat')
axs[1].plot(esantioane[indice1:indice2], x[indice1:indice2])
axs[1].set_title('semnal nefiltrat')
plt.tight_layout()
plt.savefig("generated_images/ex1_i.pdf", format="pdf")
plt.show()
# pentru semnalul filtrat se observa ca sunt mai putine componente de frecventa si mult mai putine oscilatii, pt ca am eliminat pe cele care oscilau celmai mult