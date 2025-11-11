import matplotlib.pyplot as plt
import scipy
import numpy as np
import math
import time
from scipy.io import wavfile

# N = [128, 256, 512, 1024, 2048, 4096] # 8192
#
# # 1
# t = np.linspace(0, 1, 10000)
# x = np.sin(2 * np.pi * 70 * t)
# calc_time_me = []
# calc_time_me_fft = []
# calc_time_np = []
#
# for a in N:
# # DFT
#     start = time.time()
#     X = np.zeros(a, dtype=np.complex128)
#     for m in range(a):
#         for n in range(a):
#             X[m] += x[n] * np.exp(-2 * np.pi * 1j * m * n / a)
#     end = time.time()
#     calc_time_me.append(end - start)
#
#     # FFT
#     start = time.time()
#     X = np.zeros(a, dtype=np.complex128)
#     for m in range(int(a / 2)):
#         for n in range(int(a / 2)):
#             X[m] += x[2 * m] * np.exp(-2 * np.pi * 1j * 2 * m * n / a) + x[2 * m + 1] * np.exp(
#                 -2 * np.pi * 1j * (2 * m + 1) * n / a)
#     end = time.time()
#     calc_time_me_fft.append(end - start)
#
#
#     # NP FFT
#     start = time.perf_counter() # daca ii dau cu time, face mult prea repede si da append 0
#                                     # salveaza niste valori in cache ig?
#     X = np.fft.fft(x, a)
#     end = time.perf_counter()
#     calc_time_np.append(end - start)
#
# print("My time: " + str(calc_time_me))
# print("My time fft: " + str(calc_time_me_fft))
# print("Np time: " + str(calc_time_np))
#
# fig, ax = plt.subplots()
# ax.plot(N, calc_time_me, label="Implementare de mana DFT")
# ax.plot(N, calc_time_me_fft, label="Implementare de mana FFT")
# ax.plot(N, calc_time_np, label="Implementare numpy")
# ax.set_xlabel("DFT points")
# ax.set_ylabel("Timp (s)")
# ax.legend()
# ax.grid()
# ax.set_yscale("log")
# plt.show()

# TODO; de facut calculul vectorial, nu cu cicluri for (pt ca np inmulteste matricile in c si da mult
# mai repede

# 2
# f < fs / 2, unde fs e frecv de esantionare, acum vreau ca fs / 2 < f
fs = 20
f = 8
t = np.linspace(0, 1, fs, endpoint=False)
t_good = np.linspace(0, 1, 10000, endpoint=False)

signal_real = np.sin(2 * np.pi * f * t_good)
signal_esant = np.sin(2 * np.pi * f * t)

f1 = f + fs
f2 = f + 2 * fs

signal1_good = np.sin(2 * np.pi * f1 * t_good)
signal1 = np.sin(2 * np.pi * f1 * t)

signal2_good = np.sin(2 * np.pi * f2 * t_good)
signal2 = np.sin(2 * np.pi * f2 * t)

fig, axs = plt.subplots(3, figsize=(8, 6))
axs[0].plot(t_good, signal_real)
axs[0].plot(t, signal_esant, marker="o", linestyle="")
axs[1].plot(t_good, signal1_good)
axs[1].plot(t, signal1, marker="o", linestyle="")
axs[2].plot(t_good, signal2_good)
axs[2].plot(t, signal2, marker="o", linestyle="")
plt.tight_layout()
plt.show()
#
# # 3 fs / 2 > f
# fs = 200
# f = 8
# t = np.linspace(0, 1, fs, endpoint=False)
# t_good = np.linspace(0, 1, 10000, endpoint=False)
#
# signal_real = np.sin(2 * np.pi * f * t_good)
# signal_esant = np.sin(2 * np.pi * f * t)
#
# f1 = f + fs
# f2 = f + 2 * fs
#
# signal1_good = np.sin(2 * np.pi * f1 * t_good)
# signal1 = np.sin(2 * np.pi * f1 * t)
#
# signal2_good = np.sin(2 * np.pi * f2 * t_good)
# signal2 = np.sin(2 * np.pi * f2 * t)
#
# fig, axs = plt.subplots(3, figsize=(8, 6))
# axs[0].plot(t_good, signal_real)
# axs[0].plot(t, signal_esant, marker="o", linestyle="")
# axs[1].plot(t_good, signal1_good)
# axs[1].plot(t, signal1, marker="o", linestyle="")
# axs[2].plot(t_good, signal2_good)
# axs[2].plot(t, signal2, marker="o", linestyle="")
# plt.tight_layout()
# plt.show()
# Se observa ca semnalul din prima figura este esantionat corect acum, dar pentru ca celelalte
# sunt calculate in functie de fs, inca apare fenomenul de aliasing


# 4
# Frecventele emise de contrabas sunt intre 40 si 200 de HZ
# Frecventa minima cu care trebuie esantionat semnalul trece-banda, astfel incat
# cel discretizat sa contina toate componentele pe care le poate produce trebuie sa fie macar
# dublul celei mai inalte frecvente care limiteaza banda
# Avand in vedere ca semnalul nostru nu este centrat in 0, lungimea de banda este de
# 200 - 40 = 160 Hz
# ceea ce inseamna ca
# => fs >= 2B => fs >= 320 => fsmin = 320
# daca am fi aplicat criteriul Nyquist, fs ar fi trebuit sa fie 200 * 2 = 400H Hz (intrucat
# cea mai mare frecventa ar fi fost 200), dar aici semnalul meu nu se centreaza in 0
# asa ca de asta fac scaderea 200 - 40, ca sa vad care este B


# 5
# Vocalele a, e, i au o intensitate a culorii mai puternica (adica e amplitudinea sinusoidei mai mare)
# de asemenea, vocala u este mai slaba si are frecventa mai joasa (coloana nu ajunge la fel de sus)

# 6
# sample_rate, signal = wavfile.read("vocale.wav")
# #Pentru ca semnalul este stereo, selectez doar primul canal
# if len(signal.shape) > 1:
#     signal = signal[:, 0]
#
# N = len(signal)
# group_dim = int(N * 1 / 100)
# step = group_dim // 2
# group_nr = (N - group_dim) // step + 1
# spectrogram = []
#
# for i in range(group_nr):
#     start = i * step
#     end = start + group_dim
#
#     if end > N:
#         break
#
#     group = signal[start:end]
#     fft_group = np.fft.fft(group)
#     spectrogram.append(np.abs(fft_group))
#
# spectrogram = np.array(spectrogram).T # ca sa am frecventa pe oriz
# spectrogram = spectrogram[:group_dim//2, :]
# spectrogram_db = 20 * np.log10(spectrogram + 1e-10)
#
# t = np.linspace(0, N / sample_rate, group_nr)
# freq = np.linspace(0, sample_rate / 2, group_dim // 2)
#
# plt.pcolormesh(t, freq, spectrogram_db, shading='gouraud', cmap='plasma')
#
# plt.xlabel('Timp (s)')
# plt.ylabel('Frecvență (Hz)')
# plt.colorbar(label='Magnitudine (dB)')
# plt.show()
#daca inchizi un ochi si inclini capul seamana putin cu ala din Audacity

# 7
p_signal_db = 90
snr_db = 80

# SNR = Psemnal / Pzgomot
# SNRdb = 10log10 SNR
# SNRdb = Psemnaldb - Pzgomotdb
print("Puterea zgomotului este de " + str(p_signal_db - snr_db) + " decibeli")