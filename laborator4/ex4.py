# fie frecventele semnalului f1 = 40Hz, f2 = 60Hz,
# f3=80Hz, f4 = 200Hz

# frecventa maxima = 200Hz
# de aceasta frecventa tinem cont pentru a esantiona semnalul compus fara pierderi
# aplicam teorema de esantionare Nyquist-Shannon si obtinem:
# fs > 2 * 200 > 400
# deci frecventa minima va fi 401 esantioane per sec
# daca am fi ales 381, putem fi siguri numai de pastrarea frecventelor mai mici sau egale cu 190Hz
# pentru cele mai mari se produce alierea, putand fi confundate cu frecvente mai mici