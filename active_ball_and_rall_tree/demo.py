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


def analyze_simulation(xtot, cables, t, V, window_for_autocorrel=50, recordings=1):

    if recordings==0: # means full recording
        muV_exp, sV_exp, Tv_exp = [], [], []
        
        for i in range(len(cables)): # loop over levels
            n_level = max(1,2**(i-1)) # number of levels
            for k in range(cables[i]['NSEG']): # loop over segments first
                muV_exp.append(0)
                sV_exp.append(0)
                Tv_exp.append(0)
                # they all have the same discretization
                for j in range(n_level): # loop over branches then
                    v = np.array([V[it][i][j][k] for it in range(int(.1*len(t)), len(t))]).flatten()
                    muV_exp[-1] += v.mean()/n_level
                    sV_exp[-1] += v.std()/n_level
                    v_acf, t_shift = autocorrel(v, window_for_autocorrel, (t[1]-t[0]))
                    Tv_exp[-1] += np.trapz(v_acf, t_shift)/n_level
        muV_exp, sV_exp, Tv_exp = np.array(muV_exp), np.array(sV_exp), np.array(Tv_exp)
        
    elif recordings==1: # means soma recording
        v = np.array(V[0])
        # discarding transient rise with [len(v)/5:]
        v_acf, t_shift = autocorrel(v[len(v)/5:], window_for_autocorrel, (t[1]-t[0]))
        muV_exp, sV_exp, Tv_exp = v[len(v)/5:].mean(), v[len(v)/5:].std(), np.trapz(v_acf, t_shift)

    elif recordings==2: # means soma recording
        v = np.array(V[0])
        # discarding transient rise with [len(v)/5:]
        v_acf, t_shift = autocorrel(v[len(v)/5:], window_for_autocorrel, (t[1]-t[0]))
        muV_exp, sV_exp, Tv_exp = v[len(v)/5:].mean(), v[len(v)/5:].std(), np.trapz(v_acf, t_shift)
        
    return muV_exp, sV_exp, Tv_exp


def plot_time_traces(t, V, cables, EqCylinder, title='', recordings=[[0,0,0.5]]):

    # time window
    i1, i2 = 0, min([int(1000/(t[1]-t[0])),len(t)])

    # we define the points that we want to extract

    # ---- soma
    i_level0, i_branch0, frac_seg0 = 1, 0, 0.
    i_seg0 = int(frac_seg0*cables[i_level0]['NSEG'])
    v0 = np.array([V[i][i_level0][i_branch0][i_seg0] for i in range(i1,i2)])

    # ---- medial
    if len(cables)==2:
        i_level1, i_branch1 =0, 0
    else:
        i_level1, i_branch1= len(cables)/2+1, max([1,2**(len(cables)/2)-1])
    frac_seg1 =.5
    i_seg1 = int(frac_seg1*cables[i_level1]['NSEG'])
    v1 = np.array([V[i][i_level1][i_branch1][i_seg1] for i in range(i1,i2)])

    # ---- distal
    i_level2, i_branch2, frac_seg2 = len(cables)-1, 0, 1.
    i_seg2 = int(frac_seg2*cables[i_level2]['NSEG'])-1
    v2 = np.array([V[i][i_level2][i_branch2][i_seg2] for i in range(i1,i2)])
    
    ## first we draw the fig where we will see the inputs
    fig, ax = brt_drawing.make_fig(EqCylinder, cables[1]['D'],\
        added_points=[\
                      [i_level2, i_branch2+1, frac_seg2,'r','D', 10],\
                      [i_level1, i_branch1+1, frac_seg1,'g','D', 10],\
                      [i_level0, i_branch0+1, frac_seg0,'b','D', 10]])
    fig2 = plt.figure()

    plt.title(title)
    plt.plot(1e-3*t[i1:i2], v0, 'b', label='soma')
    plt.plot(1e-3*t[i1:i2], v1, 'g', label='medial')
    plt.plot(1e-3*t[i1:i2], v2, 'r', label='distal')
    
    plt.xlabel('time (s)')
    plt.ylabel('$V_m$ (mV)')
    plt.legend(loc='best', prop={'size':'small'})
    plt.tight_layout()
    plt.savefig('./fig2.svg', format='svg')

    return fig2

def get_plotting_instructions():
    return """
args = data['args'].all()
sys.path.append('../../')
from common_libraries.graphs.my_graph import set_plot
fig, ax = plt.subplots(figsize=(5,3))
plt.subplots_adjust(left=.2,bottom=.2)
plt.plot(data['t'], data['V'][1], 'k-', label='no NMDA')
plt.plot(data['t'], data['Vnmda'][1], 'r-', label='with NMDA')
plt.plot(data['t'], data['V'][0], 'k:', label='no NMDA')
plt.plot(data['t'], data['Vnmda'][0], 'r:', label='with NMDA')
ax.legend()
set_plot(ax, xlabel='time (ms)', ylabel='$V_m$ (mV)')
"""

if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser(description=
     """ 
     description of the whole modulus here
     """
    ,formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--mean_model", help="mean model ?", action="store_true")
    parser.add_argument('-r', "--recording", type=int, help="Number of recording points: 1 (soma), 2 (soma and distal), 0 (full) ", default=20)
    parser.add_argument("--fe_prox", type=float, help="excitatory synaptic frequency in proximal compartment", default=0.7)
    parser.add_argument("--fi_prox", type=float, help="inhibitory synaptic frequency in proximal compartment", default=1.)
    parser.add_argument("--fe_dist", type=float, help="excitatory synaptic frequency in distal compartment", default=0.7)
    parser.add_argument("--fi_dist", type=float, help="inhibitory synaptic frequency in distal compartment", default=1.)
    parser.add_argument("--fi_soma", type=float, help="inhibitory synaptic frequency at soma compartment", default=1.)
    parser.add_argument("--synchrony", type=float, help="synchrony of presynaptic spikes", default=0.)
    parser.add_argument("--discret_sim", type=int, help="space discretization for numerical simulation", default=20)
    parser.add_argument("--tstop_sim", type=float, help="max simulation time (s)", default=2.)
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
    parser.add_argument("--filename", '-f', help="filename",type=str, default='data.npz')
    
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
    # shotnoise_input = {'synchrony':args.synchrony,
    #                    'fi_soma':args.fi_prox,
    #                    'fe_prox':args.fe_prox,'fi_prox':args.fi_prox,
    #                    'fe_dist':args.fe_dist,'fi_dist':args.fi_dist}
    
    if args.update_plot:
        data = dict(np.load(args.filename))
        data['plot'] = get_plotting_instructions()
        np.savez(args.filename, **data)
    else:
        from numerical_simulations.nrn_simulations import *
        nrn.nrn_load_dll('../numerical_simulations/x86_64/.libs/libnrnmech.so')
        print('Running simulation -- no NMDA [...]')
        t, V = run_simulation(shotnoise_input, cables, params, tstop=args.tstop_sim, dt=args.dt, seed=args.seed, recordings2='cable_end', recordings='soma')
        print('Running simulation -- with NMDA [...]')
        t, Vnmda = run_simulation(shotnoise_input, cables, params, tstop=args.tstop_sim, dt=args.dt, seed=args.seed, recordings2='cable_end', recordings='soma', nmda_on=True)
        np.savez(args.filename, args=args, t=t, V=V, Vnmda=Vnmda, plot=get_plotting_instructions())

    
    # muV_exp, sV_exp, Tv_exp = analyze_simulation(x_exp, cables, t, V)
    #     np.save(file, [x_exp, shotnoise_input, muV_exp, sV_exp, Tv_exp, soma, stick, params])
        
    #     # now plotting of simulated membrane potential traces
    #     plot_time_traces(t, V, cables, params['EqCylinder'],\
    #         title='$\\nu_e^p$=  %1.2f Hz, $\\nu_e^d$=  %1.2f Hz, $\\nu^p_i$= %1.2f Hz, $\\nu^d_i$= %1.2f Hz' % (args.fe_prox,args.fe_dist,args.fi_prox,args.fi_prox))
    #     plt.show()

    # # ===== now analytical calculus ========
    # try:
    #     x_exp, shotnoise_input, muV_exp, sV_exp, Tv_exp, soma, stick, params = np.load(file)
    #     # constructing the space-dependent shotnoise input for the simulation
    #     make_comparison_plot(x_th, muV_th, sV_th, Tv_th,\
    #                      x_exp, muV_exp, sV_exp, Tv_exp, shotnoise_input)    
    #     plt.show()
    # except IOError:
    #     print '======================================================'
    #     print 'no numerical data available !!!  '
    #     print 'either you run the simulation with the --simulation option !' 
    #     print 'either you provide a datafile through the --file option !' 
    #     print '======================================================'
