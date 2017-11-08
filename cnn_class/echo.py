import sys

import matplotlib.pyplot as plt
import numpy as np
import wave
from scypi.io.wavefile import write


spf = wave.open('hw.wav', 'r')

signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
print("Signal Shape: {}".format(signal.shape))

plt.plot(signal)
plt.title("HW Without Echo")
plt.show()

delta = np.array([1., 0., 0.])
noecho = np.convolve(signal, delta)
print("Noecho Shape: {}".format(noecho.shape))

noecho = noecho.astype(np.int16)
write('noecho.wav', 16000, noecho)

f = np.zeros(16000)
f[0] = 1
f[4000] = 0.6
f[8000] = 0.3
f[12000] = 0.2
f[15999] = 0.1
out = np.convolve(signal, f)

out = out.astype(np.int16)
write('echo.wav', 16000, out)