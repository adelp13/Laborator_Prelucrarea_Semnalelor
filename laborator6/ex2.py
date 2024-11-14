import numpy as np
import matplotlib.pyplot as plt

N = 7

p = np.random.rand(N)
q = np.random.rand(N)

# folosim teorema convolutiei: produsul polinoamelor = convolutie(p, q) = ifft((fft_p * fft_q))

produs_convolutie = np.convolve(p, q)
# extindem domeniul frecventei pentru ca produsul va afea 2*N-1 elemente; fft nu stie sa extinda, considera inputul multiplu de perioada
p_extins = np.concatenate([p, np.zeros(N - 1)])
q_extins = np.concatenate([q, np.zeros(N - 1)])
fft_p = np.fft.fft(p_extins)
fft_q = np.fft.fft(q_extins)
fft_pq = fft_p * fft_q

produs_fft = np.fft.ifft(fft_pq).real
print("Produsul prin convolutie:", produs_convolutie)
print("Produsul prin fft+inmultire+ifft:", produs_fft)
