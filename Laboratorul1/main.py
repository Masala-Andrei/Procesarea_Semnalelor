import numpy as np
import matplotlib.pyplot as plt


# 1.
# a) Simuleaza axa timpului printr-un sir de numere (il stochez in time_axis)
time_axis = np.linspace(0, 0.03, num=int(0.03 / 0.0005))

# b)
x = np.cos(520 * np.pi * time_axis + np.pi / 3)
y = np.cos(280 * np.pi * time_axis - np.pi / 3)
z = np.cos(120 * np.pi * time_axis + np.pi / 3)
fig, axs = plt.subplots(3)
axs[0].set_xlabel("Timp (s)")
axs[1].set_xlabel("Timp (s)")
axs[2].set_xlabel("Timp (s)")
axs[0].set_ylabel("x(t)")
axs[1].set_ylabel("y(t)")
axs[2].set_ylabel("z(t)")
axs[0].plot(time_axis, x)
axs[1].plot(time_axis, y)
axs[2].plot(time_axis, z)
for ax in axs:
    ax.set_xlim(0, 0.03)
plt.tight_layout()
plt.show()

# c)
fig2, axs2 = plt.subplots(3)
interval = np.linspace(0,0.03,200)
axs2[0].set_xlabel("TImp (s)")
axs2[1].set_xlabel("TImp (s)")
axs2[2].set_xlabel("TImp (s)")
axs2[0].stem(interval, (np.cos(520 * np.pi * interval + np.pi / 3)))
axs2[1].stem(interval, (np.cos(280 * np.pi * interval - np.pi / 3)))
axs2[2].stem(interval, (np.cos(120 * np.pi * interval + np.pi / 3)))
for ax in axs2:
    ax.set_xlim(0,0.03)
plt.tight_layout()
plt.show()

# 2.
# a) Un semnal sinusoidal de frecventa de 400 Hz, care sa contina 1600 de esantioane.
time_axis = np.linspace(0, 1, 1600)
plt.plot(time_axis, np.sin(2 * np.pi * time_axis * 400))
plt.show()

# b)
time_axis = np.linspace(0, 3, 3 * 200)
plt.stem(time_axis, np.sin(2 * np.pi * time_axis * 800))
plt.show()

# c)
freq_signal = 240
duration = 4 / 240
t = np.linspace(0, duration, 50)
plt.stem(t, freq_signal * t - np.floor(freq_signal * t))
plt.title("Semnal Sawtooth")
plt.xlabel("Timp (s)")
plt.ylabel("c[t]")
plt.grid(True)
plt.show()

# d)
freq_signal = 300
duration = 1.0
t = np.linspace(0,duration,100)
plt.stem(t, np.sign(np.sin(2 * np.pi * freq_signal * t)))
plt.title("Semnal Square")
plt.xlabel("Timp(s)")
plt.grid(True)
plt.show()

# e)
I = np.random.rand(128,128)
plt.imshow(I)
plt.title("Semnal aleator 2D")
plt.show()

# f)
I = np.zeros((128, 128))
for i in range(128):
    for j in range(128):
        if (i * j) % 7 == 0 or (i + j) % 5 == 2:
            I[i][j] = 1
plt.imshow(I)
plt.title("Semnal creat de mine 2D")
plt.show()

# 3
# a)
print("Pentru cu frecventa de esantionare de 2000 Hz , intervalul intre 2 esantioane este " + f"{1 / 2000: .4f}" + " secunde" )

# b)
print("O ora de achizitie, pentru un esantion memorat pe 4 biti, la o frecv de esantionare de 2000 Hz, timp de o ora, va ocupa " + f"{4 * 2000 * 60 * 60 / 8 / 1024}" + " MB")