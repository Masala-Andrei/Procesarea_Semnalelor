import numpy as np
import matplotlib.pyplot as plt

N = 1000
t = np.linspace(0, N, N)
trend = (3 * t ** 2 + 2 * t + 5) * 1e-5  # trb sa inmultesc cu asta ca altfel da urat pt ca e f mare
season = 2 * np.sin(t * np.pi * 0.02) + 3 * np.sin(t * np.pi * 0.07)
noise = np.random.normal(0, 1, N)
series = trend + season + noise