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
inh_factor = 5.8

fe_baseline, fi_baseline, synch_baseline = 0.2, 0.2*inh_factor, 0.05
fe_vector = np.array([0.,fe_baseline,0.4])
fi_vector = np.round(fe_vector*inh_factor,1)
synch_vector = np.array([0., synch_baseline, 0.4])

def create_set_of_exps(args):
    SET_OF_EXPS = [\
                   {'label':'$\\nu_e^p$ (Hz) \n prox. exc. ',
                    'xticks':fe_vector,
                    'fe_prox':fe_vector,
                    'fe_dist':np.ones(N_POINTS)*fe_baseline,
                    'fi_prox':np.ones(N_POINTS)*fi_baseline,
                    'fi_dist':np.ones(N_POINTS)*fi_baseline,
                    'synchrony':np.ones(N_POINTS)*synch_baseline,
                    'muV_exp':np.zeros((N_POINTS, args.SEED)), 'sV_exp':np.zeros((N_POINTS, args.SEED)), 'Tv_exp':np.zeros((N_POINTS, args.SEED))
                    },
                   {'label':'$\\nu_i^p$ (Hz) \n prox. inh. ',
                    'xticks':fi_vector,
                    'fe_prox':np.ones(N_POINTS)*fe_baseline,
                    'fe_dist':np.ones(N_POINTS)*fe_baseline,
                    'fi_prox':fi_vector,
                    'fi_dist':np.ones(N_POINTS)*fi_baseline,
                    'synchrony':np.ones(N_POINTS)*synch_baseline,
                    'muV_exp':np.zeros((N_POINTS, args.SEED)), 'sV_exp':np.zeros((N_POINTS, args.SEED)), 'Tv_exp':np.zeros((N_POINTS, args.SEED))
                    },
                   {'label':'$\\nu_e^d$ (Hz) \n distal exc. ',
                    'xticks':fe_vector,
                    'fe_prox':np.ones(N_POINTS)*fe_baseline,
                    'fe_dist':fe_vector,
                    'fi_prox':np.ones(N_POINTS)*fi_baseline,
                    'fi_dist':np.ones(N_POINTS)*fi_baseline,
                    'synchrony':np.ones(N_POINTS)*synch_baseline,
                    'muV_exp':np.zeros((N_POINTS, args.SEED)), 'sV_exp':np.zeros((N_POINTS, args.SEED)), 'Tv_exp':np.zeros((N_POINTS, args.SEED))
                    },
                   {'label':'$\\nu_i^d$ (Hz) \n distal inh.',
                    'xticks':fi_vector,
                    'fe_prox':np.ones(N_POINTS)*fe_baseline,
                    'fe_dist':np.ones(N_POINTS)*fe_baseline,
                    'fi_prox':np.ones(N_POINTS)*fi_baseline,
                    'fi_dist':fi_vector,
                    'synchrony':np.ones(N_POINTS)*synch_baseline,
                    'muV_exp':np.zeros((N_POINTS, args.SEED)), 'sV_exp':np.zeros((N_POINTS, args.SEED)), 'Tv_exp':np.zeros((N_POINTS, args.SEED))
                    },
                   {'label':'synchrony',
                    'xlabel':'synchrony',
                    'xticks':synch_vector,
                    'fe_prox':np.ones(N_POINTS)*fe_baseline,
                    'fe_dist':np.ones(N_POINTS)*fe_baseline,
                    'fi_prox':np.ones(N_POINTS)*fi_baseline,
                    'fi_dist':np.ones(N_POINTS)*fi_baseline,
                    'synchrony':synch_vector,
                    'muV_exp':np.zeros((N_POINTS, args.SEED)), 'sV_exp':np.zeros((N_POINTS, args.SEED)), 'Tv_exp':np.zeros((N_POINTS, args.SEED))
                    }]
    return SET_OF_EXPS
             

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
    params_for_cable_theory(stick, params)

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
    parser.add_argument("--seed", type=int, help="seed fo random numbers",default=37)
    parser.add_argument("--SEED", type=int, help="number of changed SEEDS",default=3)
    parser.add_argument("--discret_sim", type=int, help="space discretization for numerical simulation", default=20)
    parser.add_argument("--tstop", type=float, help="max simulation time (s)", default=3000.)
    parser.add_argument("--dt", type=float, help="simulation time step (ms)", default=0.025)
    parser.add_argument("--discret_th", type=int, help="discretization for theoretical evaluation",default=20)
    
    parser.add_argument("--MEAN_MODEL", action='store_true')
    parser.add_argument("--file",default='')

    args = parser.parse_args()

    SET_OF_EXPS = create_set_of_exps(args)
    
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

        ## MAKING THE BASELINE EXPERIMENT
        shtn_input = {'synchrony':synch_baseline,
                      'fe_prox':fe_baseline, 'fi_prox':fi_baseline,
                      'fe_dist':fe_baseline, 'fi_dist':fi_baseline}
        for s in range(args.SEED):
            print('baseline sim. , seed=', s)
            t, V = run_simulation(shtn_input, cables, params,\
                                  recordings='soma',
                                  tstop=args.tstop, dt=args.dt, seed=(ii+args.seed+s*(s+1))**3%10000)
            for EXP in SET_OF_EXPS:
                EXP['muV_exp'][int(N_POINTS/2.),s], EXP['sV_exp'][int(N_POINTS/2.),s],\
                  EXP['Tv_exp'][int(N_POINTS/2.),s] = analyze_simulation(x_exp, cables, t, V, recordings='soma')

        for EXP in SET_OF_EXPS:
            for i in np.delete(np.arange(N_POINTS), int(N_POINTS/2.)):
                shtn_input = {'synchrony':EXP['synchrony'][i],
                              'fe_prox':EXP['fe_prox'][i], 'fi_prox':EXP['fi_prox'][i],
                              'fe_dist':EXP['fe_dist'][i], 'fi_dist':EXP['fi_dist'][i]}
                for s in range(args.SEED):
                    print('sim=', i, ', seed=', s)
                    t, V = run_simulation(shtn_input, cables, params,\
                                          recordings='soma',
                                          tstop=args.tstop, dt=args.dt, seed=(ii+args.seed+s*(s+2))**3%10000)
                    EXP['muV_exp'][i,s], EXP['sV_exp'][i,s], EXP['Tv_exp'][i,s] = analyze_simulation(x_exp, cables, t, V, recordings='soma')
                                                      
                # # # keeping a trace of the full spatial profile, and the theoretical comparison
                # fig = plot_time_traces(t, V, cables, params['EqCylinder'])
                # fig.savefig('data/trace_'+str(ii)+'.svg')
                # muV_exp, sV_exp, Tv_exp = analyze_simulation(x_exp, cables, t, V, recordings='soma')
                # x_th, muV_th, sV_th, Tv_th  = \
                #   get_analytical_estimate(shtn_input,
                #                         soma, stick, params,
                #                         discret=args.discret_th)
                # fig = make_comparison_plot(x_th, muV_th, sV_th, Tv_th,\
                #          x_exp, muV_exp, sV_exp, Tv_exp, shtn_input)    
                # fig.savefig('data/spatial_profile_'+str(ii)+'.svg')
                ii+=1
                plt.close('all')
        np.save(file, [soma, stick, params, SET_OF_EXPS, args])

    import os
    if not os.path.isfile(file):
        print('--------------------> input file DOES NOT EXIST')
        print('--------------------> taking the last one !')
        file, i, flist ='', 0, os.listdir("data")
        while (file=='') and (i<1e3):
            f = flist[i]
            i+=1
            if len(f.split("vars_"))>1:
                file = 'data/'+f
        print(file)
  
    # now theoretical plot
    soma, stick, params, SET_OF_EXPS, args = np.load(file)
    Tm0 = get_membrane_time_constants(soma, stick, params)

    fig, AX = plt.subplots(3, 5, figsize=(15,10))

    # plotting all points in all plots so that they have the same boundaries !!
    for EXP in SET_OF_EXPS:
        for y, i in zip([EXP['muV_exp'], EXP['sV_exp'], EXP['Tv_exp']], list(range(3))):
            for ax in AX[i,:]:
                ax.plot(-0.+0*y, 1.1*y, 'wD', lw=0, alpha=0.)
                ax.plot(-0.+0*y, .9*y, 'wD', lw=0, alpha=0.)

    YTICKS = [[-70,-60,-50], [3,5,7], [12, 20, 28]]
    YLIM = [[-75,-40], [1.9,8.], [9,31]]
    
    for EXP, ii in zip(SET_OF_EXPS, list(range(len(SET_OF_EXPS)))):
        
        SHTN_INPUT =  {'synchrony':EXP['synchrony'],
                        'fe_prox':EXP['fe_prox'], 'fi_prox':EXP['fi_prox'],
                        'fe_dist':EXP['fe_dist'], 'fi_dist':EXP['fi_dist']}
        
        muV_th, sV_th, Tv_th, muG_th = get_the_fluct_prop_at_soma(SHTN_INPUT, params, soma, stick)
        for ax, x, y, yticks, ylim in zip(AX[:,ii], [1e3*muV_th, 1e3*sV_th, 1e3*Tm0*Tv_th],\
                                          [EXP['muV_exp'], EXP['sV_exp'], EXP['Tv_exp']],\
                                          YTICKS, YLIM):
            ax.errorbar(np.linspace(-.2,.2,len(y)), np.array(y).mean(axis=1),\
                        yerr=np.array(y).std(axis=1), marker='D', color='k')
            ax.plot(np.linspace(-.2,.2,len(x)), x, '-', color='lightgray', lw=3)
            if (ax==AX[-1,0]):
                set_plot(ax, xticks=np.linspace(-.2,.2,len(x)), xlim=[-.3,.3],\
                         xlabel=EXP['label'], yticks=yticks, ylim=ylim,
                         xticks_labels=[str(round(EXP['xticks'][i],2)) for i in range(N_POINTS)])
            elif (ax==AX[-1,ii]):
                set_plot(ax, xticks=np.linspace(-.2,.2,len(x)), yticks_labels=[], xlim=[-.3,.3],\
                         xlabel=EXP['label'], yticks=yticks, ylim=ylim, 
                         xticks_labels=[str(round(EXP['xticks'][i],2)) for i in range(N_POINTS)])
            elif ax in AX[:,0]:
                set_plot(ax, xticks=np.linspace(-.2,.2,len(x)),\
                         xticks_labels=[], xlim=[-.3,.3], yticks=yticks, ylim=ylim)
            else:
                set_plot(ax, xticks=np.linspace(-.2,.2,len(x)), yticks_labels=[],\
                         xlim=[-.3,.3], xticks_labels=[], yticks=yticks, ylim=ylim)
            
    for ax, ylabel in zip(AX[:,0], ['$\mu_V$ (mV)', '$\sigma_V$ (mV)', '$\\tau_V$ (ms)']):
        ax.set_ylabel(ylabel)

    plt.show()
    
        
