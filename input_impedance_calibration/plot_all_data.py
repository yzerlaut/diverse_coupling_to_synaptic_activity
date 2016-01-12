import numpy as np
import matplotlib.pylab as plt
import os, sys
sys.path.append('/home/yann/work/python_library/')
sys.path.append('../')
import my_graph as graph

fig, AX = plt.subplots(2, 2, figsize=(11,8))
MICE, RATS = [], []

psd_boundaries = [200,900]
all_freq, all_psd, all_phase = [np.empty(0, dtype=float) for i in range(3)]
print all_freq
HIGH_BOUND = 450 # higher nound for frequency

for file in os.listdir("./intracellular_data/"):
    if file.endswith("_rat.txt"):
        _ = 0
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
X = 300+np.arange(3)*300

for m in MICE:
    rm = m['psd'][:3].mean()
    r = (rm-psd_boundaries[0])/(psd_boundaries[1]-psd_boundaries[0])
    AX[0,0].loglog(m['freq'][m['freq']<HIGH_BOUND], m['psd'][m['freq']<HIGH_BOUND], 'D-', color=mymap(r,1), ms=4)
    AX[0,1].semilogx(m['freq'][m['freq']<HIGH_BOUND], m['phase'][m['freq']<HIGH_BOUND], 'D-', color=mymap(r,1), ms=4)

AX[0,0].annotate('n='+str(len(MICE))+' cells', (.2,.2), xycoords='axes fraction', fontsize=18)

ax2 = fig.add_axes([0.59,0.7,0.015,0.18])
cb = graph.build_bar_legend(X, ax2, mymap, scale='linear', color_discretization=100,\
                            bounds=psd_boundaries,
                            label=r'$R_\mathrm{m}$ ($\mathrm{M}\Omega$)')

AX[1,0].loglog(all_freq[all_freq<HIGH_BOUND], all_psd[all_freq<HIGH_BOUND], 'kD', ms=2, alpha=.2)
AX[1,1].semilogx(all_freq[all_freq<HIGH_BOUND], all_phase[all_freq<HIGH_BOUND], 'kD', ms=2, alpha=.2, label='data')

## THEN MODEL

from theory.analytical_calculus import * # where the core calculus lies

# common to all plots, the frequency range we will look at [0.1,1000] Hz
dt, tstop = 1.3e-3, 10.
t = np.arange(int(tstop/dt))*dt
f = rfft.time_to_freq(len(t), dt)

# somatic parameters
soma = {'L': 7*1e-6, 'D': 10*1e-6, 'NSEG': 1, 'exc_density':1e9, 'inh_density':1e9, 'name':'soma'}

# stick parameters
stick = {'L': 250*1e-6, 'D': .6*1e-6, 'B':5, 'NSEG': 30, 'exc_density':1e9, 'inh_density':1e9, 'name':'dend'}

# branching properties
EqCylinder = np.linspace(0, 1, stick['B']) # equally space branches ! UNITLESS, multiplied only in the func by stick['L']

# biophysical properties
params = {'g_pas': 2.5e-5*1e4, 'cm' : 1.*1e-2, 'Ra' : 100.*1e-2, 'El': -65e-3,
          'Qe' : 1.e-9 , 'Te' : 5.e-3, 'Ee': 0e-3,\
          'Qi' : 1.5e-9 , 'Ti' : 5.e-3, 'Ei': -80e-3,
          'Ee': 0e-3, 'Ei': -80e-3}

def get_input_imped(soma, stick, params, EqCylinder1):
    EqCylinder2 = EqCylinder1*stick['L'] # NOW REAL EqCylinder !!
    params_for_cable_theory(stick, params) # setting cable membrane constants
    output = get_the_input_impedance_at_soma(f, EqCylinder2, soma, stick, params)
    psd, phase = np.abs(output)/10e6, (np.angle(output)+np.pi/2.)%(2.*np.pi)-np.pi/2.
    return psd, phase

### MEAN MODEL
psd, phase = get_input_imped(soma, stick, params, EqCylinder)
AX[1,0].loglog(f, psd, 'k-', alpha=.8, lw=4)
AX[1,1].semilogx(f, -phase, 'k-', alpha=.8, lw=4, label='mean model')

### MODEL VARIATIONS
N=5
L_soma = 0*np.linspace(-1,1,N)*1e-6
D_soma = 0*np.linspace(-5,5,N)*1e-6
L_dend = np.linspace(-90,90,N)*1e-6
D_dend = np.linspace(-.25,.25,N)*1e-6
for ls, ds, ld, dd, r in zip(L_soma, D_soma, L_dend, D_dend, np.linspace(1,0,N)):
    soma1, stick1, params1 = soma.copy(), stick.copy(), params.copy()
    soma1['L'] += ls
    soma1['D'] += ds
    stick1['L'] += ld
    stick1['D'] += dd
    psd, phase = get_input_imped(soma1, stick1, params1, EqCylinder)
    AX[1,0].loglog(f, psd, '-', color=mymap(r,1), ms=5)
    AX[1,1].semilogx(f, -phase, '-', color=mymap(r,1), ms=5)


    
### ===================== FINAL graph settings

AX[1,1].legend(frameon=False, prop={'size':'xx-small'})

ax3 = fig.add_axes([0.59,0.23,0.015,0.18])
cb = graph.build_bar_legend([0,1], ax3, mymap, scale='linear', color_discretization=100,\
                            bounds=psd_boundaries,ticks_labels=['',''],
                            label='size variations')

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




