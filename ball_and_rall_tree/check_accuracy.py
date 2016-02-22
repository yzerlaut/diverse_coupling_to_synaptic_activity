import numpy as np
import matplotlib.pylab as plt
import sys, time
sys.path.append('../code')
from my_graph import set_plot
import fourier_for_real as rfft
sys.path.append('../')
from theory.analytical_calculus import *
from data_firing_response.analyze_data import get_Rm_range
from input_impedance_calibration.get_calib import adjust_model_prop
from demo import *

from firing_response_description.template_and_fitting import final_func

N_POINTS = 3

fe_vector = np.array([.7,1.,1.3])
fi_vector = np.array([.7,1.,1.3])*7.
synchrony_vector = np.array([0., 0.3, 0.6])

SET_OF_EXPS = [\
               {'label':'increasing prox. exc. \n $\\nu_e^p$ (Hz)',
                'xticks':fe_vector,
                'fe_prox':fe_vector,
                'fe_dist':np.ones(N_POINTS)*fe_vector[1],
                'fi_prox':np.ones(N_POINTS)*fi_vector[1],
                'fi_dist':np.ones(N_POINTS)*fi_vector[1],
                'synchrony':np.ones(N_POINTS)*synchrony_vector[1],
                'muV_exp':np.zeros(N_POINTS), 'sV_exp':np.zeros(N_POINTS), 'Tv_exp':np.zeros(N_POINTS)
                },
               {'label':'increasing distal exc. \n $\\nu_e^d$ (Hz)',
                'xticks':fe_vector,
                'fe_prox':np.ones(N_POINTS)*fe_vector[1],
                'fe_dist':fe_vector,
                'fi_prox':np.ones(N_POINTS)*fi_vector[1],
                'fi_dist':np.ones(N_POINTS)*fi_vector[1],
                'synchrony':np.ones(N_POINTS)*synchrony_vector[1],
                'muV_exp':np.zeros(N_POINTS), 'sV_exp':np.zeros(N_POINTS), 'Tv_exp':np.zeros(N_POINTS)
                },
               {'label':'increasing prox. inh. \n $\\nu_i^p$ (Hz)',
                'xticks':fi_vector,
                'fe_prox':np.ones(N_POINTS)*fe_vector[1],
                'fe_dist':np.ones(N_POINTS)*fe_vector[1],
                'fi_prox':fi_vector,
                'fi_dist':np.ones(N_POINTS)*fi_vector[1],
                'synchrony':np.ones(N_POINTS)*synchrony_vector[1],
                'muV_exp':np.zeros(N_POINTS), 'sV_exp':np.zeros(N_POINTS), 'Tv_exp':np.zeros(N_POINTS)
                },
               {'label':'increasing distal inh. \n $\\nu_i^d$ (Hz)',
                'xticks':fe_vector,
                'fe_prox':np.ones(N_POINTS)*fe_vector[1],
                'fe_dist':np.ones(N_POINTS)*fe_vector[1],
                'fi_prox':np.ones(N_POINTS)*fi_vector[1],
                'fi_dist':fi_vector,
                'synchrony':np.ones(N_POINTS)*synchrony_vector[1],
                'muV_exp':np.zeros(N_POINTS), 'sV_exp':np.zeros(N_POINTS), 'Tv_exp':np.zeros(N_POINTS)
                },
               {'label':'increasing synchrony',
                'xlabel':'synchrony',
                'xticks':synchrony_vector,
                'fe_prox':np.ones(N_POINTS)*fe_vector[1],
                'fe_dist':np.ones(N_POINTS)*fe_vector[1],
                'fi_prox':np.ones(N_POINTS)*fi_vector[1],
                'fi_dist':np.ones(N_POINTS)*fi_vector[1],
                'synchrony':synchrony_vector,
                'muV_exp':np.zeros(N_POINTS), 'sV_exp':np.zeros(N_POINTS), 'Tv_exp':np.zeros(N_POINTS)
                }]
             

def get_model(args):

    soma = {'L': args.L_soma*1e-6, 'D': args.D_soma*1e-6, 'NSEG': 1,\
            'exc_density':1e9, 'inh_density':(1e-5)**2/20., 'name':'soma'}

    # baseline stick parameters, will be modified by geometry
    stick = {'L': args.L_stick*1e-6, 'D': args.D_stick*1e-6, 'B':args.branches, 'NSEG': args.discret_sim,\
             'exc_density':(1e-5)**2/30., 'inh_density':(1e-5)**2/6., 'name':'dend'}

    # biophysical properties
    params = {'g_pas': 1e-4*1e4, 'cm' : 1.*1e-2, 'Ra' : 200.*1e-2}

    # now synaptic properties
    params['Qe'], params['Qi'] = args.Qe*1e-9, args.Qi*1e-9
    params['Te'], params['Ti'] = 5e-3, 5e-3
    params['Ee'], params['Ei'] = 0e-3, -80e-3
    params['El'] = -65e-3#0e-3, -80e-3
    params['fraction_for_L_prox'] = 5./6.
    params['factor_for_distal_synapses_weight'] = 2.
    params['factor_for_distal_synapses_tau'] = 1.

    return soma, stick, params

if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser(description=
     """ 
     description of the whole modulus here
     """
    ,formatter_class=argparse.RawTextHelpFormatter)
    
    # ball and stick properties
    parser.add_argument("--L_stick", type=float, help="Length of the stick in micrometer", default=1000.)
    parser.add_argument("--L_prox_fraction", type=float, help="fraction of tree corresponding to prox. compartment", default=0.)
    parser.add_argument("--D_stick", type=float, help="Diameter of the stick", default=2.)
    parser.add_argument("--L_soma", type=float, help="Length of the soma in micrometer", default=10.)
    parser.add_argument("--D_soma", type=float, help="Diameter of the soma in micrometer", default=10.)
    parser.add_argument("-B", "--branches", type=int, help="Number of branches (equally spaced)", default=1)
    # synaptic properties
    parser.add_argument("--Qe", type=float, help="Excitatory synaptic weight (nS)", default=1.)
    parser.add_argument("--Qi", type=float, help="Inhibitory synaptic weight (nS)", default=3.)


    parser.add_argument("--SIM", action='store_true') # flag for running simuluation !
    parser.add_argument("--seed", type=int, help="seed fo random numbers",default=3)
    parser.add_argument("--discret_sim", type=int, help="space discretization for numerical simulation", default=20)
    parser.add_argument("--tstop_sim", type=float, help="max simulation time (s)", default=2.)
    parser.add_argument("--dt", type=float, help="simulation time step (ms)", default=0.025)
    parser.add_argument("--discret_th", type=int, help="discretization for theoretical evaluation",default=20)
    
    parser.add_argument("--MEAN_MODEL", action='store_true')
    parser.add_argument("--file",default='')

    args = parser.parse_args()

    if args.file=='':
        file = 'data/vars_'+time.strftime("%d.%m.%Y_%H.%M")+'.npy'
    else:
        file = args.file
    
    if args.SIM:

        if args.MEAN_MODEL:
            soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')
        else:
            soma, stick, params = get_model(args)

        x_exp, cables = setup_model(soma, stick, params)    
        ii=0
        for EXP in SET_OF_EXPS:
            for i in range(N_POINTS):
                shtn_input = {'synchrony':EXP['synchrony'][i],
                              'fe_prox':EXP['fe_prox'][i], 'fi_prox':EXP['fi_prox'][i],
                              'fe_dist':EXP['fe_dist'][i], 'fi_dist':EXP['fi_dist'][i]}
                t, V = run_simulation(shtn_input, cables, params,\
                                      tstop=args.tstop_sim*1e3, dt=args.dt, seed=(ii+args.seed)**3)
                muV_exp, sV_exp, Tv_exp = analyze_simulation(x_exp, cables, t, V)

                EXP['muV_exp'][i], EXP['sV_exp'][i], EXP['Tv_exp'][i] = muV_exp[0], sV_exp[0], Tv_exp[0]
                                                      
                # # keeping a trace of the full spatial profile, and the theoretical comparison
                fig = plot_time_traces(t, V, cables, params['EqCylinder'])
                fig.savefig('data/trace_'+str(ii)+'.svg')
                muV_exp, sV_exp, Tv_exp = analyze_simulation(x_exp, cables, t, V)
                x_th, muV_th, sV_th, Tv_th  = \
                  get_analytical_estimate(shtn_input,
                                        soma, stick, params,
                                        discret=args.discret_th)
                fig = make_comparison_plot(x_th, muV_th, sV_th, Tv_th,\
                         x_exp, muV_exp, sV_exp, Tv_exp, shtn_input)    
                fig.savefig('data/spatial_profile_'+str(ii)+'.svg')
                ii+=1
                plt.close()
        np.save(file, [soma, stick, params, SET_OF_EXPS])

    # now theoretical plot
    soma, stick, params, SET_OF_EXPS = np.load(file)
    Tm0 = get_membrane_time_constants(soma, stick, params)

    fig, AX = plt.subplots(3, figsize=(15,10))
    
    Xticks, Xticks_Labels = np.zeros(0), []
    for EXP, ii in zip(SET_OF_EXPS, range(len(SET_OF_EXPS))):
        SHTN_INPUT =  {'synchrony':EXP['synchrony'],
                        'fe_prox':EXP['fe_prox'], 'fi_prox':EXP['fi_prox'],
                        'fe_dist':EXP['fe_dist'], 'fi_dist':EXP['fi_dist']}
        get_the_fluct_prop_at_soma(SHTN_INPUT, params, soma, stick)
        muV_th, sV_th, Tv_th, muG_th = get_the_fluct_prop_at_soma(SHTN_INPUT, params, soma, stick)
        for ax, x, y in zip(AX, [1e3*muV_th, 1e3*sV_th, 1e3*Tm0*Tv_th],\
                            [EXP['muV_exp'], EXP['sV_exp'], EXP['Tv_exp']]):
            ax.plot(np.linspace(-.2,.2,len(y))+ii, y, 'kD')
            ax.plot(np.linspace(-.2,.2,len(x))+ii, x, '-', color='lightgray', lw=3)
            Xticks = np.concatenate([Xticks, np.linspace(-.2,.2,len(x))+ii])
            for i in range(len(EXP['muV_exp'])):
                Xticks_Labels.append(str(round(EXP['xticks'][i],1)))
            
    for ax, ylabel in zip(AX, ['$\mu_V$ (mV)', '$\sigma_V$ (mV)', '$\\tau_V$ (ms)']):
        set_plot(ax, ylabel=ylabel, xticks=[])
    set_plot(ax, ylabel=ylabel, xticks=Xticks, xticks_labels=Xticks_Labels)

    plt.show()
    
        
