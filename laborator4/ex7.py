import numpy as np

putere_semnal_DB = 90 #dB
SNR_DB = 80 #dB

# avem formulele:
# SNR = putere_semnal / putere_zgomot
# SNR_DB = 10log10(SNR)
# => SNR = 10^(SNR_DB/10)
# => putere_semnal / putere_zgomot = 10^(SNR_DB/10)
# => putere_zgomot = putere_semnal / (10**(SNR_DB / 10) )

# dar putere_semnal trebuie sa fie in unitati liniare, nu in dB
# conversie:
putere_semnal_liniar = 10 ** (putere_semnal_DB/10)
putere_zgomot_liniar = putere_semnal_liniar / (10**(SNR_DB / 10) )
print("Puterea liniara a zgomotului este: ", putere_zgomot_liniar)
# raspuns: 10