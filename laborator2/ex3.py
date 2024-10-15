import sounddevice as sd
import scipy.io.wavfile as wf
import numpy as np
import matplotlib.pyplot as plt

fs = 44100  # frecventa de esantionare si de redare

#a: semnal sinusoidal de frecventa 400Hz, care sa contina 1600 esantioane
frecventa_semnal = 400
t = np.linspace(0, 1, fs)
a = np.sin(2 * np.pi * t * frecventa_semnal + np.pi/3)

sd.play(a, fs)
sd.wait() # daca nu asteptam codul va merge mai departe si se va termina fara sa permita ascultarea semnalelor

a_int16 = np.int16(a * 32767) # trebuie convertit din float 64 pt a putea fi redat

wf.write('generated_sounds/ex3_semnal_a.wav', fs, a_int16)
rate, a_incarcat = wf.read('generated_sounds/ex3_semnal_a.wav')
sd.play(a_incarcat, fs)
sd.wait()
#b: Un semnal sinusoidal de frecventa 800 Hz, care sa dureze 3 secunde
frecventa_semnal = 800
durata_semnal = 3
t = np.linspace(0, durata_semnal, int(durata_semnal * fs)) # am crescut nr de esantioane fata de lab trecut
b = np.sin(2 * np.pi * frecventa_semnal * t + np.pi/16)

b_int16 = np.int16(b * 32767)
wf.write('generated_sounds/ex3_semnal_b.wav', fs, b_int16)

sd.play(b, fs)
sd.wait()

#c: Un semnal de tip sawtooth de frecventa 240 Hz (puteti folosi functiile numpy.floor sau numpy.mod)
# trebuie o functie care urca si apoi se intoarce la valoarea initiala
frecventa_semnal = 240
t = np.linspace(0, 1, fs) # frecventa de esantioanre = frecventa de redare, altfel se poate modifica durata sunetului
c = t * frecventa_semnal - np.floor(t * frecventa_semnal)

sd.play(c, fs)
sd.wait()

#d: Un semnal de tip square de frecventa 300 Hz (puteti folosi functia numpy.sign)
t = np.linspace(0, 1, fs)
frecventa_semnal = 300
d = np.sign(np.cos(2 * np.pi * frecventa_semnal * t))

sd.play(d, fs)
sd.wait()