import numpy as np
import matplotlib.pyplot as plt

frecventa_esantionare = 1000
perioada_esantionare = 1 / frecventa_esantionare

durata_semnal = 0.05
frecventa_semnal = 35
t = np.linspace(0, durata_semnal, int(durata_semnal * frecventa_esantionare))
sinus_initial = 3 * np.sin(2 * frecventa_semnal * np.pi * t)

sinus_decimat_a = sinus_initial[::4]
t_decimat_a = t[::4]
sinus_decimat_b = sinus_initial[1::4]
t_decimat_b = t[1::4]

# pastrand doar esantioanele cu indicii 0 4 8 ... (subpunctul a) respectiv 1 5 9 ... (b) scadem frecventa de esantionare de 4 ori
# f = 1000 si f/4 = 250 sunt suficiente pentru a pastra informatiile de baza despre semnalul cu frecventa 35
# 35 << 1000
# dar cele 2 semnale decimate a si b nu au pantele sinusului la fel de fine ca semnalul initial (sunt colturoase)
# de aceea a difera de b si se observa clar de la ce esantion am plecat cu decimarea
# la frecvente de esantionare foarte mari fata de f_semnal nu ar fi vizibila prea usor diferenta dintre a si b, si poate nici intre semnalul initial si cel decimat

#daca semnalul nedecimat ar avea frecventa 245 (sau o valoare comparabila cu frecventa de esantionare 1000, nu cum era 35),
#semnalul nedecimat poate fi afisat fara pierderi, dar frecventa decimata ar fi 250 = 5 + frecventa_semnal
# deci semnalele a si b nu pot fi afisate fara pierderi ca cel nedecimat
# (acest lucru incepe de la f_semnal_nedecimat = 126 in sus, pt ca f_esantionare/4 = 250 = 2 * 125)
# ele vor parea ca sunt niste semnale cu frecvente mai mici decat cel intial si ca a = b + faza

fis, axs = plt.subplots(3)
fis.suptitle(f"frecventa_semnal_initial={frecventa_semnal}, frecventa_esantionare={frecventa_esantionare}")
axs[0].plot(t, sinus_initial)
#axs[0].stem(t, sinus_initial)
axs[0].set_title("sinus initial")
axs[1].plot(t_decimat_a, sinus_decimat_a)
#axs[1].stem(t_decimat_a, sinus_decimat_a)
axs[1].set_title("sinus decimat a)")
axs[2].plot(t_decimat_b, sinus_decimat_b)
#axs[2].stem(t_decimat_b, sinus_decimat_b)
axs[2].set_title("sinus decimat b (de pe pozitia 1)")
plt.tight_layout()
plt.show()
