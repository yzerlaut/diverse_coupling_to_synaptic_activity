import numpy as np
import matplotlib.pylab as plt
import os, sys
sys.path.append('/home/yann/work/python_library/')
import my_graph as graph

data = np.loadtxt('sample.txt', unpack=True)

t = data[0]-data[0][0]

vm, I = data[1], data[3]

fig, ax = plt.subplots(2, figsize=(4,2))
plt.subplots_adjust(left=.22)

ax[0].plot(t, I, 'k-')
ax[0].plot([0,1], [0,0], 'k-', lw=4)
ax[0].annotate('1s', (.1,.8), xycoords='axes fraction')
graph.set_plot(ax[0], ['left'], ylabel='$I$ (pA)', xticks=[])

ax[1].plot(t, vm, 'k-')
graph.set_plot(ax[1], ['left'], ylabel='$V_m$ (mV)', xticks=[])

from scipy.optimize import curve_fit

# first episode
t0 = t[t<3.4]
f1 = 0.584
sin1 = lambda t, phi, A, El: A*np.sin(2.*np.pi*f1*t+phi)+El
y = sin1(t0, 0., 4., -60)
popt, pcov = curve_fit(sin1, t0, vm[:len(t0)])
ax[1].plot(t0, sin1(t0, *popt), 'r-')

cond = (t>4.8)
t1 = t[cond]
f1 = 1.268
sin1 = lambda t, phi, A, El: A*np.sin(2.*np.pi*f1*t+phi)+El
y = sin1(t0, 0., 4., -60)
popt, pcov = curve_fit(sin1, t[cond], vm[cond])
ax[1].plot(t[cond], sin1(t[cond], *popt), 'r-')


fig.savefig('fig.svg', format='svg')



