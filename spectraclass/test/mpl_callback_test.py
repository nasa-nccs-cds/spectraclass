import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 1.0, 0.01)
s = np.sin(2 * np.pi * t)
fig, ax = plt.subplots()
ax.plot(t, s)
canvas = ax.figure.canvas

def on_move(event):
    if event.inaxes:
        print('data coords %f %f' % (event.xdata, event.ydata))

canvas.mpl_connect('motion_notify_event', on_move)


plt.show()