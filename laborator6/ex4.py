import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

# a
semnal = np.genfromtxt('kaggle/Train.csv', delimiter=',', skip_header=1, usecols=-1, dtype=int)
fs = 1/3600
# un esantion pe ora => pt 3 zile = 72 esantioane
N = 72
x = semnal[:N]

# b
dimensiuni_fereastra = [5, 9, 13, 17]

plt.figure()
plt.title("ex4_b filtru tip medie alunecatoare")
t = np.linspace(0, N - 1, N)
plt.plot(t, x, label="semnal initial")

for w in dimensiuni_fereastra:
    semnal_netezit = np.convolve(x, np.ones(w), "valid") / w
    # fiecare esantion (mai putin primele si ultimele w - 1) e media celor w esantioane din stanga lui (inculsiv el)
    t = np.linspace(0, N - 1, len(semnal_netezit))
    plt.plot(t, semnal_netezit, label=f"dimensiune fereastra={w}")

plt.legend()
plt.savefig('generated_images/ex4_b.pdf', format="pdf")
plt.show()
# cu cat fereastra e mai mare cu atat rezultatul este mai uniform

# c
frecventa_Nyquist = fs / 2
# frecventa nyquist = 1 / 7200
# nu putem cauta si elimina frecvente mai mari de atat pentru ca esantionarea nu permite detectarea lor.
# observam comportamente periodice si in interiorul unei zile,
# deci trebuie sa pastram si perioade mai mici de 24 de ore, nu doar de la o zi in sus, plus ca in grafic analizam doar 3 zile, nu luni sau ani.

# cele mai mici perioade importante:
# corespund unor componente de frecventa continute in cea de o zi si care apar de ex la pranz cand este mult trafic.
# acestea au perioada de cateva ore (esantioane) si prezenta lor in semnal este cea mai mare la anumite ore din zi.
# se vede si in grafic cum noaptea sunt cele mai putine masini (de ex esantioanele 2-7 adica din prima zi).
# deci putem sa luam frecvente cu perioada minima de 4.5 ore,
# fiind o perioada suficient de mare pentru a fi considerat un eveniment semnificativ dintr-o zi

# frecvete cu perioada >= 4.5 ore => frecventa = 1 / 4.5 * 3600 = 1 / 16200 Hz
frecventa_taiere = 1 / 16200
frecv_taiere_normalizata = frecventa_taiere / frecventa_Nyquist
#print(frecv_taiere_normalizata)
# valoarea normalizata este 0.44 din f Nyquist

# d
ordin_filtru = 5
rp = 5
b_butter, a_butter = sc.signal.butter(ordin_filtru, frecv_taiere_normalizata, btype='low')
b_cheby1, a_cheby1 = sc.signal.cheby1(ordin_filtru, rp, frecv_taiere_normalizata, btype='low')

# e
plt.figure()
plt.title("ex4_e butterworth vs chebyshev1(rp 5) de ordine 5")
t = np.linspace(0, N - 1, N)
plt.plot(t, x, label="semnal initial")

semnal_butter = sc.signal.filtfilt(b_butter, a_butter, x)
t = np.linspace(0, N - 1, len(semnal_butter))
plt.plot(t, semnal_butter, label=f"butterworth")

semnal_cheby1 = sc.signal.filtfilt(b_cheby1, a_cheby1, x)
t = np.linspace(0, N - 1, len(semnal_cheby1))
plt.plot(t, semnal_cheby1, label=f"chebyshev")

plt.legend()
plt.savefig('generated_images/ex4_e.pdf', format="pdf")
plt.show()

# butterworth: componentele de frecventa sub 0.44 raman aproape la fel (raspuns plat), de la 0.44 in sus scade amplitudinea treptat, nu brusc.
# deci nu e sigur ca am eliminat frecventele de peste 0.44, dar stim ca semnalul sub 0.4 este intact.
# se observa si in plot cum butter seamana mai mult cu originalul decat cheby pentru ca a diminuat doar ce e zgomot;
# (acest lucru arata si daca frecventa de taiere pentru zgomot a fost aproximata bine)

# chebyshev in schimb nu pastreaza intact semnalul sub 0.44 (rp influenteaza cat de mult oscileaza),
# dar asta permite o banda de tranzitie mai scurta, adica mai putine frecvente de peste 0.44 care raman.

# in practica nu putem filtra la milimetru (ar fi nevoie de un ordin infinit, adica sa ne
# uitam la un infinit de vecini, deci e nevoie de un compromis timp-frecventa).
# deci va fi nevoie de o tranzitie de la frecventele pastrate la cele eliminate,
# ceea ce inseamna un interval de frecvente care au amplitudine partiala in loc sa fie intreaga sau nula
# 1.tranzitie prezenta mai mult sub 0.44 (cheby), frecventele de peste 0.44 eliminate aproape complet, sau 2.tranzitie mai mult peste 0.44 (butter)
# aceste 2 filtre par la extremele acestui compromis

# in cazul de fata: scopul era sa eliminam zgomotul. pot fi frecvente de zgomot destul de aproape de 0.44, 0.44 insemnand 4.5 ore.
# o perioada de 4 sau 3 ore este prea mica pentru a o considera comportament periodic pe parcursul unei zile => zgomot
# zgomotul fiind aproape de 0.44 am putea alege chebishev, chiar daca micosram amplitudinea unor frecvente putin sub 0.44 (tot le vom putea detecta)
# asa ne vom asigura ca orice componenta de frecventa gasita nu este zgomot

# f
# ambele filtre cu un ordin mai mic si unul mai mare:
fig, axs = plt.subplots(2, 1, figsize=(10, 6))
t = np.linspace(0, N - 1, N)
axs[0].plot(t, x, label="semnal initial")
axs[1].plot(t, x, label="semnal initial")
axs[0].set_title("butter")
axs[1].set_title("cheby rp=5")
ordine = [1, 11]

for ordin in ordine:
    b_butter, a_butter = sc.signal.butter(ordin, frecv_taiere_normalizata, btype='low')
    b_cheby1, a_cheby1 = sc.signal.cheby1(ordin, rp, frecv_taiere_normalizata, btype='low')

    semnal_butter = sc.signal.filtfilt(b_butter, a_butter, x)
    t = np.linspace(0, N - 1, len(semnal_butter))
    axs[0].plot(t, semnal_butter, label=f"butter ordin={ordin}")

    semnal_cheby1 = sc.signal.filtfilt(b_cheby1, a_cheby1, x)
    t = np.linspace(0, N - 1, len(semnal_cheby1))
    axs[1].plot(t, semnal_cheby1, label=f"cheby ordin={ordin}")

axs[0].legend()
axs[1].legend()
plt.subplots_adjust(top=1.0)
plt.suptitle("diverse ordine pt butter si cheby1(rp=5)")
plt.tight_layout()
plt.savefig('generated_images/ex4_f_ordine_diferite.pdf', format="pdf")
plt.show()

# cheby1 de ordin 1, 5, 7 cu rp 0.5, 4, 10:
ordine = [1, 3, 7]
rps = [0.5, 3, 12]
fig, axs = plt.subplots(3, 1, figsize=(9, 6))
t = np.linspace(0, N - 1, N)
axs[0].plot(t, x, label="semnal initial")
axs[1].plot(t, x, label="semnal initial")
axs[2].plot(t, x, label="semnal initial")
axs[0].set_title(f"cheby oridn={ordine[0]}")
axs[1].set_title(f"cheby ordin={ordine[1]}")
axs[2].set_title(f"cheby ordin={ordine[2]}")

for i in range(len(ordine)):
    for rp in rps:
        b_cheby1, a_cheby1 = sc.signal.cheby1(ordine[i], rp, frecv_taiere_normalizata, btype='low')
        semnal_cheby1 = sc.signal.filtfilt(b_cheby1, a_cheby1, x)
        t = np.linspace(0, N - 1, len(semnal_cheby1))
        axs[i].plot(t, semnal_cheby1, label=f"cheby rp={rp}")

axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.subplots_adjust(top=1.0)
plt.suptitle("cheby1 cu diverse ordine si rp")
plt.tight_layout()
plt.savefig('generated_images/ex4_f_rp_diferite.pdf', format="pdf")
plt.show()

# un ordin mai mare are banda de tranzitie mai mica pentru ca analizeaza mai bine semnalul.
# semnalele cu filtru de ordin mai mare sunt si mai uniforme pentru ca
# delimiteaza mai precis frecventele de trecere si nu avem prea multe frecvente ramase cu amplitudini micsorate.
# deci forma semnalului este mai apropiata de ce ne dorim, dar nu si pozitia esantioanelor, pentru ca se pot produce intarzieri, adica un eveniment se decaleaza.
# in grafic intarzierea se observa mai mult la chebyshev, dar nu in toate esantioanele.

# la butter diferentele intre ordine nu sunt la fel de vizibile nici ca forma a semnalului, nici ca intarziere,
# deci alegem un ordin mic, tinand cont si ca un ordin mare creste complexitatea.
# deci mai ales pentru chebyshev ar fi mai potrivit un ordin mic spre mediu (3-4), plus ca aici putem regla si cu rp.

# rp inseamna diferenta dintre oscilatiile benzii de trecere;
# cu cat rp e mai mare cu atat tranzitia e mai rapida si frecventele sunt delimitate mai bine
# totusi, o tranzitie prea brusca de la rp sau ordin prea mare elimina prea mult din semnal, tinzand spre un semnal constant,
# acesta e alt motiv pentru care ordinul nu trebuie sa fie prea mare, pe langa complexitate si intarzieri

# rp=3 pare sa micsoreze destul intervalul de frecvente din cele de trecere carora li se scade amplitudinea
# si e destul de mic cat sa nu se piarda prea mult din semnal.

# cel mai potrivit pentru trafic: chebishev ordin 3 rp 3
# (pastreaza destul de bine momentele din zi cand e trafic minim sau maxim si creeaza un semnal lin,
# eliminand interpolarile colturoase generate de un numar mic de esantioane).