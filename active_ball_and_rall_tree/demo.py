import numpy as np
import matplotlib
import matplotlib.pylab as plt
import sys, time
sys.path.append('../code/')
from my_graph import set_plot
import fourier_for_real as rfft
from signanalysis import autocorrel
sys.path.append('../')
from theory import brt_drawing
from theory.analytical_calculus import *
from scipy.optimize import curve_fit
from numerical_simulations.nrn_simulations import *
nrn.nrn_load_dll('../numerical_simulations/x86_64/.libs/libnrnmech.so')
nrn.nrn_load_dll('../numerical_simulations/nrnmech.dll')

# we calculate the parameters to plug into cable theory

# ------- model parameters in SI units ------- # 
params = {
    'g_pas': 1e-4*1e4, 'cm' : 1*1e-2, 'Ra' : 35.4*1e-2, 'El': -65e-3,\
    'Qe' : 1.e-9 , 'Te' : 5.e-3, 'Ee': 0e-3,\
    'Qi' : 1.5e-9 , 'Ti' : 5.e-3, 'Ei': -80e-3,\
    'seed' : 0}
    
FACTOR_FOR_DENSITY = 1. # because Ball & Sticks sucks !!!

# synaptic density  area (m2) per one synapse !!
# ball (soma)
soma = {'L': 20*1e-6, 'D': 20*1e-6, 'NSEG': 1,
        'exc_density':FACTOR_FOR_DENSITY*1e9,
        'inh_density':FACTOR_FOR_DENSITY*25*1e-12,
        'Ke':1e-9, 'Ki':10., 'name':'soma'}
# stick
stick = {'L': 2000*1e-6, 'D': 4*1e-6, 'NSEG': 30,\
         'exc_density':FACTOR_FOR_DENSITY*17*1e-12,
         'inh_density':FACTOR_FOR_DENSITY*100*1e-12,
         'Ke':100, 'Ki':100., 'name':'dend'}


def analyze_simulation(t, V, Vnmda, Vhh, Vall):
    i0 = min([len(t)-2,int(100/(t[1]-t[0]))]) # discarding 100ms
    # subthreshold
    muV_soma, sV_soma, Tv_soma = [], [], []
    muV_dend, sV_dend, Tv_dend = [], [], []
    for v in [V, Vnmda, Vhh, Vall]:
        muV_soma.append(np.array(v[0])[i0:].mean())
        sV_soma.append(np.array(v[0])[i0:].std())
        v_acf, t_shift = autocorrel(np.array(v[0])[i0:], 50, (t[1]-t[0]))
        Tv_soma.append(np.trapz(v_acf, t_shift))
        muV_dend.append(np.array(v[1])[i0:].mean())
        sV_dend.append(np.array(v[1])[i0:].std())
        v_acf, t_shift = autocorrel(np.array(v[1])[i0:], 50, (t[1]-t[0]))
        Tv_dend.append(np.trapz(v_acf, t_shift))
    # spikes
    vhh, vall = np.array(Vhh[0]), np.array(Vall[0])
    print(t[(vhh[1:]>10) & (vhh[:-1]<10)])
    print(len(t[(vhh[1:]>10) & (vhh[:-1]<10)]))
    Fhh = len(t[(vhh[1:]>10) & (vhh[:-1]<10)])/t[-1]
    print(1e3*Fhh)
    Fall = len(t[(vall[1:]>10) & (vall[:-1]<10)])/t[-1]
    # return muV_soma, sV_soma, Tv_soma, muV_dend, sV_dend, Tv_dend, Fhh, Fall
    return muV_soma, sV_soma, Tv_soma, Fhh, Fall


def run_all(args, shotnoise_input, cables, params, synchronous_stim, seed=0):        
    print('Running simulation -- Passive [...]')
    t, V = run_simulation(shotnoise_input, cables, params, tstop=args.tstop, dt=args.dt, seed=args.seed+seed, recordings2='cable_end', recordings='soma', synchronous_stim=synchronous_stim)
    print('Running simulation -- NMDA [...]')
    t, Vnmda = run_simulation(shotnoise_input, cables, params, tstop=args.tstop, dt=args.dt, seed=args.seed+seed, recordings2='cable_end', recordings='soma', nmda_on=True, synchronous_stim=synchronous_stim)
    print('Running simulation -- Ca2+ + NMDA [...]')
    t, Vhh = run_simulation(shotnoise_input, cables, params, tstop=args.tstop, dt=args.dt, seed=args.seed+seed, recordings2='cable_end', recordings='soma', nmda_on=False, Ca_spikes_on=False, HH_on=True, synchronous_stim=synchronous_stim)
    print('Running simulation -- Ca2+ + NMDA [...]')
    t, Vall = run_simulation(shotnoise_input, cables, params, tstop=args.tstop, dt=args.dt, seed=args.seed+seed, recordings2='cable_end', recordings='soma', nmda_on=True, Ca_spikes_on=True, HH_on=True, synchronous_stim=synchronous_stim)
    return t, V, Vnmda, Vhh, Vall

def get_plotting_instructions():
    return """
args = data['args'].all()
sys.path.append('../../')
from common_libraries.graphs.my_graph import set_plot
fig, AX = plt.subplots(2, 1, figsize=(8,6))
plt.subplots_adjust(left=.2,bottom=.2, hspace=.3, wspace=.3)
AX[0].plot(data['t'], data['V'][1], 'k-', label='passive')
AX[0].plot(data['t'], data['Vnmda'][1], 'r-', label='NMDA')
AX[0].legend(prop={'size':'xx-small'})
AX[1].plot(data['t'], data['V'][0], 'k-', label='Control (passive + AMPA + GABA)')
AX[1].plot(data['t'], data['Vnmda'][0], 'r-', label='Control + NMDA')
AX[1].plot(data['t'], 1e3*data['Vhh'][0], 'b-', label='Control + HH (soma)')
AX[1].plot(data['t'], 1e3*data['Vall'][0], 'g--', alpha=.3, lw=3, label='Control + NMDA + HH (soma) + Ca spikes')
AX[1].legend(prop={'size':'xx-small'})
if args.with_synch_stim:
    tt=args.DT_synch_stim
    while tt<args.tstop:
        AX[0].plot(tt, [-30], 'kv', ms=10)
        AX[1].plot(tt, [40], 'kv', ms=10)
        tt+=args.DT_synch_stim
set_plot(AX[0], xlabel='time (ms)', ylabel='$V_m$ (mV)')
set_plot(AX[1], xlabel='time (ms)', ylabel='$V_m$ (mV)')
"""

if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser(description=
     """ 
     description of the whole modulus here
     """
    ,formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--mean_model", help="mean model ?", action="store_true")
    parser.add_argument("--with_synch_stim", help="with synchronous stimulation ?", action="store_true")
    parser.add_argument("--N_synch_stim", type=int, help="Number of synchronous psp events", default=30)
    parser.add_argument("--DT_synch_stim", type=float, help="space between synch events", default=200)
    parser.add_argument('-r', "--recording", type=int, help="Number of recording points: 1 (soma), 2 (soma and distal), 0 (full) ", default=20)
    parser.add_argument("--fe_prox", type=float, help="excitatory synaptic frequency in proximal compartment", default=0.)
    parser.add_argument("--fi_prox", type=float, help="inhibitory synaptic frequency in proximal compartment", default=0.)
    parser.add_argument("--fe_dist", type=float, help="excitatory synaptic frequency in distal compartment", default=0.)
    parser.add_argument("--fi_dist", type=float, help="inhibitory synaptic frequency in distal compartment", default=0.)
    parser.add_argument("--fi_soma", type=float, help="inhibitory synaptic frequency at soma compartment", default=0.)
    parser.add_argument("--synchrony", type=float, help="synchrony of presynaptic spikes", default=0.)
    parser.add_argument("--discret_sim", type=int, help="space discretization for numerical simulation", default=20)
    parser.add_argument("--tstop", type=float, help="max simulation time (s)", default=2.)
    parser.add_argument("--dt", type=float, help="simulation time step (ms)", default=0.025)
    parser.add_argument("--discret_th", type=int, help="discretization for theoretical evaluation",default=20)
    parser.add_argument("--seed", type=int, help="seed fo random numbers",default=3)
    # ball and stick properties
    parser.add_argument("--L_soma", type=float, help="Length of the soma in micrometer", default=10.)
    parser.add_argument("--D_soma", type=float, help="Diameter of the soma in micrometer", default=10.)
    parser.add_argument("--L_stick", type=float, help="Length of the stick in micrometer", default=1000.)
    parser.add_argument("--L_prox_fraction", type=float, help="fraction of tree corresponding to prox. compartment",\
                        default=1.)
    parser.add_argument("--D_stick", type=float, help="Diameter of the stick", default=2.)
    parser.add_argument("-B", "--branches", type=int, help="Number of branches (equally spaced)", default=1)
    # synaptic properties
    parser.add_argument("--Qe", type=float, help="Excitatory synaptic weight (nS)", default=1.)
    parser.add_argument("--Qi", type=float, help="Inhibitory synaptic weight (nS)", default=1.5)

    parser.add_argument("-u", "--update_plot", help="plot the figures", action="store_true")
    parser.add_argument( '-f', "--filename",help="filename",type=str, default='data.npz')
    
    args = parser.parse_args()

    if args.mean_model:
        soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')
        stick['NSEG'] = 20
    else:
        # setting up the stick properties
        stick['L'] = args.L_stick*1e-6
        stick['D'] = args.D_stick*1e-6
        stick['B'] = args.branches
        stick['NSEG'] = args.discret_sim
        params['EqCylinder'] = np.linspace(0, 1, stick['B']+1)*stick['L'] # equally space branches !

        # fraction L_prox
        params['fraction_for_L_prox'] = args.L_prox_fraction

        # settign up the synaptic properties
        params['Qe'] = args.Qe*1e-9
        params['Qi'] = args.Qi*1e-9
        params['factor_for_distal_synapses_tau'] = 1.
        params['factor_for_distal_synapses_weight'] = 2.

    print(' first we set up the model [...]')
    x_exp, cables = setup_model(soma, stick, params)    

    x_stick = np.linspace(0,args.L_stick*1e-6, args.discret_sim+1) # then :
    x_stick = .5*(x_stick[1:]+x_stick[:-1])
    
    # constructing the space-dependent shotnoise input for the simulation

    inh_factor = 5.8
    fe_baseline, fi_baseline, synch_baseline = 0.1, 0.1*inh_factor, 0.05
    shotnoise_input = {'synchrony':synch_baseline,
                       'fe_prox':fe_baseline, 'fi_prox':fi_baseline,
                       'fe_dist':2.*fe_baseline, 'fi_dist':fi_baseline}
    for x, y in zip(\
        ['synchrony', 'fi_soma', 'fi_prox', 'fe_prox','fi_dist', 'fe_dist'],\
        [args.synchrony,args.fi_prox,args.fi_prox,args.fe_prox,\
         args.fi_dist,args.fe_dist]):
        if y!=0:
            shotnoise_input[x] = y
            
    if args.with_synch_stim:
        spikes = np.array([])
        tt=args.DT_synch_stim
        while tt<args.tstop:
            spikes = np.concatenate([spikes,tt+np.ones(args.N_synch_stim)])
            tt+=args.DT_synch_stim
        synchronous_stim={'location':[4,14], 'spikes':spikes}
    else:
        synchronous_stim={'location':[4,14], 'spikes':[]}
        
    if args.update_plot:
        data = dict(np.load(args.filename))
        data['plot'] = get_plotting_instructions()
        np.savez(args.filename, **data)
    else:
        t, V, Vnmda, Vhh, Vall = run_all(args, shotnoise_input, cables, params, synchronous_stim)
        print(analyze_simulation(t, V, Vnmda, Vhh, Vall))
        np.savez(args.filename, args=args, t=t, V=V, Vnmda=Vnmda, Vhh=Vhh, Vall=Vall, plot=get_plotting_instructions())
