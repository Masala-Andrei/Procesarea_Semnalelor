import itertools
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import sounddevice

# frecv de esantionare de 44,1k

axis = np.linspace(0, 1.0, 200)
x = 3 * np.sin(2 * np.pi * 30 * axis + np.pi / 2)
y = 3 * np.cos(2 * np.pi * 30 * axis)
fig, axs = plt.subplots(3)
axs[0].plot(axis, x)
axs[0].plot(axis, y)
axs[1].set_title("Sin")
axs[1].plot(axis, x)
axs[2].set_title("Cos")
axs[2].plot(axis, y)
for ax in axs:
    ax.set_xlim(0, 1)
plt.show()

2
frecventa = 2
x1 = np.sin(2 * np.pi * frecventa * axis + np.pi / 6)
x2 = np.sin(2 * np.pi * frecventa * axis + np.pi / 3)
x3 = np.sin(2 * np.pi * frecventa * axis + np.pi / 8)
x4 = np.sin(2 * np.pi * frecventa * axis + 3 * np.pi / 4)
fig, ax = plt.subplots(1)
ax.plot(axis, x1)
ax.plot(axis, x2)
ax.plot(axis, x3)
ax.plot(axis, x4)
ax.set_xlim(0, 1)
plt.show()
X = [x1, x2, x3, x4]

SNR = [0.1, 1, 10, 100]
noise = np.random.normal(0, 1, 200)
fig, ax = plt.subplots(1)
fig, axs = plt.subplots(4)
i = 0
for snr, x in zip(SNR, X):
    gamma = np.linalg.norm(x) / (snr * np.linalg.norm(noise))
    noisy_signal = x + gamma * noise
    ax.plot(axis, noisy_signal)
    axs[i].plot(axis, noisy_signal)
    axs[i].set_title("Noise " + str(snr))
    i += 1
ax.set_xlim(0, 1.0)
for ax in axs:
    ax.set_xlim(0, 1)
plt.show()

# 3
rate = 44100
time_axis = np.linspace(0, 1, rate)
signal = np.sin(2 * np.pi * time_axis * 400)
sounddevice.play(np.sin(2 * np.pi * time_axis * 400), rate)
sounddevice.wait()
scipy.io.wavfile.write('semnal1.wav', rate, signal)
rate, x = scipy.io.wavfile.read('semnal1.wav')
sounddevice.play(x, rate)
sounddevice.wait()

time_axis = np.linspace(0, 3, rate)
sounddevice.play(np.sin(2 * np.pi * time_axis * 800), rate)
sounddevice.wait()

freq_signal = 240
t = np.linspace(0, 1, rate)
sounddevice.play(freq_signal * t - np.floor(freq_signal * t), rate)
sounddevice.wait()

freq_signal = 300
duration = 1.0
t = np.linspace(0,duration,rate)
sounddevice.play(np.sign(np.sin(2 * np.pi * freq_signal * t)), rate)
sounddevice.wait()


# 4
axis = np.linspace(0,1,200)
sig1 = np.sin(2 * np.pi * 100 * axis)
sig2 = np.sin(20 * axis - np.floor(axis * 20))
fig, ax = plt.subplots(3)
ax[0].plot(axis, sig1)
ax[1].plot(axis, sig2)
ax[2].plot(axis, sig1 + sig2)
plt.show()







