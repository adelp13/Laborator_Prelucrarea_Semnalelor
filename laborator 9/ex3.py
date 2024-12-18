import numpy as np
import matplotlib.pyplot as plt

def CalcularePredictii(serie_timp, q, termeni_eroare): # predictie pentru un singur element in viitor
    serie_timp = serie_timp[-q:]
    medie = np.mean(serie_timp)
    serie_timp_centrata = serie_timp - medie

    ferestre_eroare = [termeni_eroare[i:i + q] for i in range(q)]

    # M * theta = N
    thetas, _, _, _ = np.linalg.lstsq(ferestre_eroare, serie_timp_centrata, rcond=None)
    predicție = np.dot(ferestre_eroare[-1], thetas) + medie

    return predicție

N = 1000
a = 10
b = 0.1
c = 0.2
t = np.linspace(0, 1, N)

trend = np.zeros(N)
trend = a * t * t + b * t + c

sezon = 3 * np.sin(2 * np.pi * t * 10) + 2 * np.sin(2 * np.pi * t * 80)
variatii_mici = np.random.normal(0, 0.8, N)
serie_timp = trend + sezon + variatii_mici

# aplicam MA pentru a prezice seria de timp (cu exceptia primilor p termeni)
# pentru a determina teta (coeficientii termenilor de eroare) avem in vedere ultimii q termeni din serie
# MA nu mai are m si p, doar q = p = m

qs = [5]
predictii_serie_timp = serie_timp.copy()

for index_q in range(len(qs)):
    termeni_eroare = np.random.normal(0, 1, 2*qs[index_q])
    for i in range(qs[index_q], N):
        serie_timp_decupata = serie_timp[:i]
        predictii_serie_timp[i] = CalcularePredictii(serie_timp_decupata, qs[index_q], termeni_eroare)


plt.figure()
plt.title("MA ex3")
plt.plot(t, serie_timp, label="serie timp")
plt.plot(t, predictii_serie_timp, label="predicții")
plt.xlabel("esantioane")
plt.ylabel("ampitudine")
plt.legend()
plt.savefig('generated_images/ex3.pdf', format="pdf")
plt.show()


