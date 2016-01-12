import numpy as np
import matplotlib.pylab as plt
import os, sys
sys.path.append('/home/yann/work/python_library/')
import my_graph as graph

fig, AX = plt.subplots(2, 2, figsize=(11,8))
MICE, RATS = [], []

psd_boundaries = [200,900]
all_freq, all_psd, all_phase = [np.empty(0, dtype=float) for i in range(3)]
print all_freq
HIGH_BOUND = 500 # higher nound for frequency

for file in os.listdir("./intracellular_data/"):
    if file.endswith("_rat.txt"):
        freq, psd, phase = np.loadtxt("./intracellular_data/"+file, unpack=True)
        RATS.append({'freq':freq[np.argsort(freq)], 'psd':psd[np.argsort(freq)], 'phase':(phase[np.argsort(freq)]%(2.*np.pi)+np.pi/2.)%(2.*np.pi)-np.pi/2.})
    elif file.endswith(".txt"):
        freq, psd, phase = np.loadtxt("./intracellular_data/"+file, unpack=True)
        MICE.append({'freq':freq[np.argsort(freq)], 'psd':psd[np.argsort(freq)], 'phase':(phase[np.argsort(freq)]%(2.*np.pi)+np.pi/2.)%(2.*np.pi)-np.pi/2.})
        # print file, MICE[-1]['phase'].max()
        all_freq = np.concatenate([all_freq, MICE[-1]['freq']])
        all_psd = np.concatenate([all_psd, MICE[-1]['psd']])
        all_phase = np.concatenate([all_phase, MICE[-1]['phase']])
        
    psd_boundaries[0] = min([psd_boundaries[0], psd.max()])
    psd_boundaries[1] = max([psd_boundaries[1], psd.max()])
        
np.save('full_data.npy',\
    [all_freq[all_freq<HIGH_BOUND], all_psd[all_freq<HIGH_BOUND], all_phase[all_freq<HIGH_BOUND]])


mymap = graph.get_linear_colormap()
X = 200+np.arange(3)*400 #[100,200,400,600,800,1000]

for m in MICE:
    rm = m['psd'][:3].mean()
    r = (rm-psd_boundaries[0])/(psd_boundaries[1]-psd_boundaries[0])
    AX[0,0].loglog(m['freq'][m['freq']<HIGH_BOUND], m['psd'][m['freq']<HIGH_BOUND], 'D-', color=mymap(r,1), ms=4)
    AX[0,1].semilogx(m['freq'][m['freq']<HIGH_BOUND], m['phase'][m['freq']<HIGH_BOUND], 'D-', color=mymap(r,1), ms=4)
    
# for m in RATS:
#     rm = m['psd'][:3].mean()
#     r = (m['psd'].max()-psd_boundaries[0])/(psd_boundaries[1]-psd_boundaries[0])
#     AX[1,0].loglog(m['freq'], m['psd']/rm, 'o-', color=mymap(r,1), ms=7., alpha=.5)
#     AX[0,0].loglog(m['freq'], m['psd'], 'o-', color=mymap(r,1), ms=7., alpha=.5)
#     # AX[0,0].loglog(m['freq'], m['psd'], 'o-', color='g', ms=5)
#     AX[1,1].semilogx(m['freq'], m['phase'], 'o-', color=mymap(r,1), ms=5)
#     AX[0,1].semilogx(m['freq'], m['phase'], 'o-', color='g', ms=5)

# fig2, ax2 = plt.subplots(1, figsize=(2,5))
# plt.subplots_adjust(right=.5)
ax2 = fig.add_axes([0.59,0.7,0.015,0.18])
cb = graph.build_bar_legend(X, ax2, mymap, scale='linear', color_discretization=100,\
                            bounds=psd_boundaries,
                            label=r'$R_\mathrm{m}$ ($\mathrm{M}\Omega$)')

# graph settings

# AX[0,0].loglog(MICE[-1]['freq'], MICE[-1]['psd'], 'D-', color='m', ms=5, label='mice P8-P12')
# AX[0,0].loglog(RATS[-1]['freq'], RATS[-1]['psd'], 'o-', color='g', ms=5, label='rat P12')
# AX[0,0].legend(loc='best', frameon=False)

for ax in [AX[0,0], AX[1,0]]:
    graph.set_plot(ax, xlim=[.08,1000], ylim=[6., 1200],\
               xticks=[1,10,100,1000], yticks=[10,100,1000],yticks_labels=['10','100','1000'],\
               ylabel='input impedance ($\mathrm{M}\Omega$)')

for ax in [AX[0,1], AX[1,1]]:
    graph.set_plot(ax, xlim=[.08,1000], ylim=[-.2,3.7],\
               xticks=[1,10,100,1000], yticks=[0,np.pi/2.,np.pi],\
               yticks_labels=[0,'$\pi/2$', '$\pi$'],
               xlabel='frequency (Hz)', ylabel='phase shift (Rd)')

plt.show()




