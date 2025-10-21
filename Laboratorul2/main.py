import matplotlib.pyplot as plt
import numpy as np
import sounddevice

# frecv de esantionare de 44,1k

axis = np.linspace(0, 1.0, 200)
# x = 3 * np.sin(2 * np.pi * 30 * axis + np.pi / 2)
# y = 3 * np.cos(2 * np.pi * 30 * axis)
# fig, axs = plt.subplots(3)
# axs[0].plot(axis, x)
# axs[0].plot(axis, y)
# axs[1].set_title("Sin")
# axs[1].plot(axis, x)
# axs[2].set_title("Cos")
# axs[2].plot(axis, y)
# for ax in axs:
#     ax.set_xlim(0, 1)
# plt.show()
#
# # 2
# frecventa = 2
# x1 = np.sin(2 * np.pi * frecventa * axis + np.pi / 6)
# x2 = np.sin(2 * np.pi * frecventa * axis + np.pi / 3)
# x3 = np.sin(2 * np.pi * frecventa * axis + np.pi / 8)
# x4 = np.sin(2 * np.pi * frecventa * axis + 3 * np.pi / 4)
# fig, ax = plt.subplots(1)
# ax.plot(axis, x1)
# ax.plot(axis, x2)
# ax.plot(axis, x3)
# ax.plot(axis, x4)
# ax.set_xlim(0, 1)
# plt.show()
# X = [x1, x2, x3, x4]
#
# SNR = [0.1, 1, 10, 100]
# noise = np.random.normal(0, 1, 200)
# fig, ax = plt.subplots(1)
# fig, axs = plt.subplots(4)
# i = 0
# for snr, x in zip(SNR, X):
#     gamma = np.linalg.norm(x) / (snr * np.linalg.norm(noise))
#     noisy_signal = x + gamma * noise
#     ax.plot(axis, noisy_signal)
#     axs[i].plot(axis, noisy_signal)
#     axs[i].set_title("Noise " + str(snr))
#     i += 1
# ax.set_xlim(0, 1.0)
# for ax in axs:
#     ax.set_xlim(0, 1)
# plt.show()
#
# # 3
# rate = 44100
# time_axis = np.linspace(0, 1, rate)
# signal = np.sin(2 * np.pi * time_axis * 400)
# sounddevice.play(np.sin(2 * np.pi * time_axis * 400), rate)
# sounddevice.wait()
# scipy.io.wavfile.write('semnal1.wav', rate, signal)
# rate, x = scipy.io.wavfile.read('semnal1.wav')
# sounddevice.play(x, rate)
# sounddevice.wait()
#
# time_axis = np.linspace(0, 3, rate)
# sounddevice.play(np.sin(2 * np.pi * time_axis * 800), rate)
# sounddevice.wait()
#
# freq_signal = 240
# t = np.linspace(0, 1, rate)
# sounddevice.play(freq_signal * t - np.floor(freq_signal * t), rate)
# sounddevice.wait()
#
# freq_signal = 300
# duration = 1.0
# t = np.linspace(0, duration, rate)
# sounddevice.play(np.sign(np.sin(2 * np.pi * freq_signal * t)), rate)
# sounddevice.wait()
#
# # 4
# axis = np.linspace(0, 1, 200)
# sig1 = np.sin(2 * np.pi * 100 * axis)
# sig2 = np.sin(20 * axis - np.floor(axis * 20))
# fig, ax = plt.subplots(3)
# ax[0].plot(axis, sig1)
# ax[1].plot(axis, sig2)
# ax[2].plot(axis, sig1 + sig2)
# plt.show()

# 5
# rate = 44100
# axis = np.linspace(0, 1.0, rate)
# sig1 = np.sin(2 * np.pi * axis * 520)
# sig2 = np.sin(2 * np.pi * axis * 400)
# firstSecond = np.concatenate((sig1, sig2))
# sounddevice.play(firstSecond, rate)
# sounddevice.wait()
# nu se intampla nimic special imo, doar dau play unul dupa altu

# 6
# sample_freq = 200
# axis = np.linspace(0, 1.0, sample_freq)
# sig1 = np.sin(2 * np.pi * axis * sample_freq / 2)
# sig2 = np.sin(2 * np.pi * axis * sample_freq / 4)
# sig3 = np.sin(2 * np.pi * axis * 0)  # adica ar trebui sa fie o linie?
# fig, axs = plt.subplots(3)
# axs[0].plot(axis, sig1)
# axs[1].plot(axis, sig2)
# axs[2].plot(axis, sig3)
#
# axs[0].set_ylabel("sig1")
# axs[1].set_ylabel("sig2")
# axs[2].set_ylabel("sig3")
# axs[2].set_xlabel("Time (s)")
#
# for ax in axs:
#     ax.set_xlim(0, 1)
# plt.show()

# 7
# axis = np.linspace(0, 1.0, 1000)
# axis_decimat = axis[1::4]
# axis_decimat2 = axis[2::4]
# sig_normal = np.sin(2 * np.pi * axis * 100)
# sig_decimat = np.sin(2 * np.pi * axis_decimat * 100)
# sig_decimat2 = np.sin(2 * np.pi * axis_decimat2 * 100)
#
# fig, axs = plt.subplots(3)
#
# axs[0].plot(axis, sig_normal)
# axs[0].set_ylabel("Sig Normal")
# axs[1].plot(axis_decimat, sig_decimat)
# axs[1].set_ylabel("Sig Decimat")
# axs[2].plot(axis_decimat2, sig_decimat2)
# axs[2].set_ylabel("Sig Decimat")
# plt.show()
# Semnalul decimat incepand cu al 2 lea element pare sa preia partea de jos a sinusoidei predominant, in timp ce primul
# preia partea de sus


# 8
alpha = np.linspace(-np.pi / 2, np.pi / 2, 1000)
sinus = np.sin(alpha)
sinus_aproximat = alpha
sinus_pade = (alpha - 7 * alpha ** 3 / 60) / (1 + alpha ** 2 / 20)
eroare_aprox = np.abs(sinus - sinus_aproximat)
eroare_pade = np.abs(sinus - sinus_pade)

fig, axs = plt.subplots(3)
axs[0].plot(alpha, sinus)
axs[1].plot(alpha, sinus_aproximat)
axs[2].plot(alpha, sinus_pade)
plt.show()

fig, axs = plt.subplots(2)
axs[0].plot(alpha, eroare_aprox)
axs[1].plot(alpha, eroare_pade)
plt.show()

valori = [0.1, 0.001, 0.025, 0.0003, 0.00014]
for val in valori:
    sin = np.sin(val)
    aprox = val
    pade = (val - 7 * val ** 3 / 60) / (1 + val ** 2 / 20)

    err_lin = abs(sin - aprox)
    err_pade = abs(sin - pade)

    print(f"αlpha = {val:.8f}")
    print(f"Sin: {sin:.12f}")
    print(f"Liniar: {aprox:.12f} (eroare: {err_lin:.2e})")
    print(f"Padé: {pade:.12f} (eroare: {err_pade:.2e})")
    print("-----------------------------")

