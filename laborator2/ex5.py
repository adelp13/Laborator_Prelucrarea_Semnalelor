import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf

frecventa1 = 200
frecventa2 = 1000
fs = 44100
durata_semnal = 4
t = np.linspace(0, durata_semnal, durata_semnal * fs)
sinus1 = np.sin(2 * np.pi * frecventa1 * t) # ambele sin, deci aceeasi forma de unda
sinus2 = np.sin(2 * np.pi * frecventa2 * t)

sinus_concatenat = np.concatenate((sinus1, sinus2)) # daca adunam nu ar fi concatenat, ci ar fi facut ca la exercitiul 4

sinus_concatenat_int16 = np.int16(sinus_concatenat * 32767)
wf.write('generated_sounds/ex5_semnal_concatenat.wav', fs, sinus_concatenat_int16)

sd.play(sinus_concatenat, fs) # redam pe secunda cu aceeasi frecventa cu care esantionam, asa conservam sunetul si durata sa
sd.wait()

# primul sunet se aude de parca ar vibra
# al doilea sunet este mult mai ascutit si putin mai deranjant la auz, da senzatia ca ar dura mai putin, chiar daca ambele au 4 secunde
