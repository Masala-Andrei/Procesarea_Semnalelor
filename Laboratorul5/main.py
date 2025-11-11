import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas
from datetime import datetime

x_csv = pandas.read_csv("Train.csv")
x = np.genfromtxt("Train.csv", delimiter=",", skip_header=1)
N = len(x)
# # 1a)
# # Nr de masini a fost masurat din ora in ora, ceea ce inseamna ca a fost esantionat o data la 3600
# # de secunde
# fs = 1 / 3600
#
# # b)
# start = datetime.strptime(x_csv["Datetime"].iloc[1], "%d-%m-%Y %H:%M")
# end = datetime.strptime(x_csv["Datetime"].iloc[-1], "%d-%m-%Y %H:%M")
# print(end - start)
#
# # c)
# # Frecventa maxima, daca semnalul este esaantionat corect, este jumatate din frecventa de esantionare
# max_freq = fs / 2
#
# # d)
# f = fs * np.linspace(0, N // 2, N // 2) / N
#
X = np.fft.fft(x[:, 2])
X = abs(X / N)
X = X[: (N // 2)]
# print(X)
#
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(f, X)
# ax.set_yscale("log")
# plt.show()
#
# # e)
# fig, ax = plt.subplots(figsize=(12,6))
# ax.plot(f, X)
# ax.set_title("Verificare componenta continua")
# plt.show()
# # Pentru 0, modulul transformatei are o valoare f mare
#
# # f)
# max_freq = np.argsort(X)[-5:]
# max_freq = max_freq[:-1]
# print(f[max_freq])
# perioade_zile = 1 / f[max_freq] / 3600 / 24
# print(perioade_zile)

# g)
start = x_csv.index[x_csv["Datetime"] == "01-07-2013 00:00"].tolist()[0]
end = x_csv.index[x_csv["Datetime"] == "01-08-2013 00:00"].tolist()[0]

x_luna = x[start:end, 2]
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_luna)
ax.set_xlabel("Timp [h]")
ax.set_ylabel("Nr. masini")
plt.show()

# h)
# O metoda analizand semnalul timp, ar putea fi reprezentat de faptul ca in timpul saptamanii
# fluxul de masini este mai ridicat, fata de weekend. In plus, in apropierea sarbatorilor (Craciun
# Anul Nou etc.) ar fi mai ridicat, ceea ce ar ajuta determinarea perioadei din an in care este evnimentul
# - in timpul scolii, fluxul de masini este mai ridicat si asta ar fi un indicator daca
# data este in sezonul vacantelor sau nu
# Acuratetea ar fi afectata de fluctuatii de tipul: in timpul scolii si in tp vacantelor ar putea
# aparea acelasi flux de masini, pot fi zile ploioase in care ne asteptam ca nr de masini sa fie
# mai ridicat

# i)
# In loc de a evalua o data pe ora, am putea o data la 6 ore

