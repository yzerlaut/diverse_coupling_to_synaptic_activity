import numpy as np
import matplotlib.pylab as plt
import sys, time


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
                    'muV_exp_ctrl':np.zeros((N_POINTS, args.SEED)), 'sV_exp_ctrl':np.zeros((N_POINTS, args.SEED)), 'Tv_exp_ctrl':np.zeros((N_POINTS, args.SEED)),
                    'muV_exp_nmda':np.zeros((N_POINTS, args.SEED)), 'sV_exp_nmda':np.zeros((N_POINTS, args.SEED)), 'Tv_exp_nmda':np.zeros((N_POINTS, args.SEED)),
                    'Fout_hh':np.zeros((N_POINTS, args.SEED)), 'Fout_all':np.zeros((N_POINTS, args.SEED))
                    },
                   {'label':'$\\nu_i^p$ (Hz) \n prox. inh. ',
                    'xticks':fi_vector,
                    'fe_prox':np.ones(N_POINTS)*fe_baseline,
                    'fe_dist':np.ones(N_POINTS)*fe_baseline,
                    'fi_prox':fi_vector,
                    'fi_dist':np.ones(N_POINTS)*fi_baseline,
                    'synchrony':np.ones(N_POINTS)*synch_baseline,
                    'muV_exp_ctrl':np.zeros((N_POINTS, args.SEED)), 'sV_exp_ctrl':np.zeros((N_POINTS, args.SEED)), 'Tv_exp_ctrl':np.zeros((N_POINTS, args.SEED)),
                    'muV_exp_nmda':np.zeros((N_POINTS, args.SEED)), 'sV_exp_nmda':np.zeros((N_POINTS, args.SEED)), 'Tv_exp_nmda':np.zeros((N_POINTS, args.SEED)),
                    'Fout_hh':np.zeros((N_POINTS, args.SEED)), 'Fout_all':np.zeros((N_POINTS, args.SEED))
                    },
                   {'label':'$\\nu_e^d$ (Hz) \n distal exc. ',
                    'xticks':fe_vector,
                    'fe_prox':np.ones(N_POINTS)*fe_baseline,
                    'fe_dist':fe_vector,
                    'fi_prox':np.ones(N_POINTS)*fi_baseline,
                    'fi_dist':np.ones(N_POINTS)*fi_baseline,
                    'synchrony':np.ones(N_POINTS)*synch_baseline,
                    'muV_exp_ctrl':np.zeros((N_POINTS, args.SEED)), 'sV_exp_ctrl':np.zeros((N_POINTS, args.SEED)), 'Tv_exp_ctrl':np.zeros((N_POINTS, args.SEED)),
                    'muV_exp_nmda':np.zeros((N_POINTS, args.SEED)), 'sV_exp_nmda':np.zeros((N_POINTS, args.SEED)), 'Tv_exp_nmda':np.zeros((N_POINTS, args.SEED)),
                    'Fout_hh':np.zeros((N_POINTS, args.SEED)), 'Fout_all':np.zeros((N_POINTS, args.SEED))
                    },
                   {'label':'$\\nu_i^d$ (Hz) \n distal inh.',
                    'xticks':fi_vector,
                    'fe_prox':np.ones(N_POINTS)*fe_baseline,
                    'fe_dist':np.ones(N_POINTS)*fe_baseline,
                    'fi_prox':np.ones(N_POINTS)*fi_baseline,
                    'fi_dist':fi_vector,
                    'synchrony':np.ones(N_POINTS)*synch_baseline,
                    'muV_exp_ctrl':np.zeros((N_POINTS, args.SEED)), 'sV_exp_ctrl':np.zeros((N_POINTS, args.SEED)), 'Tv_exp_ctrl':np.zeros((N_POINTS, args.SEED)),
                    'muV_exp_nmda':np.zeros((N_POINTS, args.SEED)), 'sV_exp_nmda':np.zeros((N_POINTS, args.SEED)), 'Tv_exp_nmda':np.zeros((N_POINTS, args.SEED)),
                    'Fout_hh':np.zeros((N_POINTS, args.SEED)), 'Fout_all':np.zeros((N_POINTS, args.SEED))
                    },
                   {'label':'synchrony',
                    'xlabel':'synchrony',
                    'xticks':synch_vector,
                    'fe_prox':np.ones(N_POINTS)*fe_baseline,
                    'fe_dist':np.ones(N_POINTS)*fe_baseline,
                    'fi_prox':np.ones(N_POINTS)*fi_baseline,
                    'fi_dist':np.ones(N_POINTS)*fi_baseline,
                    'synchrony':synch_vector,
                    'muV_exp_ctrl':np.zeros((N_POINTS, args.SEED)), 'sV_exp_ctrl':np.zeros((N_POINTS, args.SEED)), 'Tv_exp_ctrl':np.zeros((N_POINTS, args.SEED)),
                    'muV_exp_nmda':np.zeros((N_POINTS, args.SEED)), 'sV_exp_nmda':np.zeros((N_POINTS, args.SEED)), 'Tv_exp_nmda':np.zeros((N_POINTS, args.SEED)),
                    'Fout_hh':np.zeros((N_POINTS, args.SEED)), 'Fout_all':np.zeros((N_POINTS, args.SEED))
                    }]
    return SET_OF_EXPS
             
def get_plotting_instructions():
    return """
args = data['args'].all()
ii=0
from check_accuracy import *
SET_OF_EXPS = create_set_of_exps(args)
for EXP in SET_OF_EXPS:
    EXP['muV_exp_ctrl'], EXP['sV_exp_ctrl'], EXP['Tv_exp_ctrl'],\
       EXP['muV_exp_nmda'], EXP['sV_exp_nmda'], EXP['Tv_exp_nmda'], EXP['Fout_hh'], EXP['Fout_all'] = np.load(args.filename+str(ii)+'.npy')
    ii+=1
sys.path.append('../../')
from common_libraries.graphs.my_graph import set_plot
fig, AX = plt.subplots(4, 5, figsize=(10,8))
plt.subplots_adjust(left=.25, bottom=.25, wspace=.4, hspace=.4)
# plotting all points in all plots so that they have the same boundaries !!
for EXP in SET_OF_EXPS:
    for y, i in zip([EXP['muV_exp_ctrl'], EXP['sV_exp_ctrl'], EXP['Tv_exp_ctrl'], EXP['Fout_hh']], list(range(4))):
        for ax in AX[i,:]:
            ax.plot(-0.+0*y, 1.1*y, 'wD', lw=0, alpha=0.)
            ax.plot(-0.+0*y, .9*y, 'wD', lw=0, alpha=0.)
YTICKS = [[-70,-60,-50], [3,5,7], [12, 20, 28], [0,10,20]]
YLIM = [[-75,-30], [1.9,10.], [9,45], [-2,30]]
for EXP, ii in zip(SET_OF_EXPS, list(range(len(SET_OF_EXPS)))):
    for ax, y1, x, yticks, ylim, col in zip(AX[:,ii],\
                                      [EXP['muV_exp_ctrl'], EXP['sV_exp_ctrl'], EXP['Tv_exp_ctrl'], EXP['Fout_hh']],\
                                      [EXP['muV_exp_nmda'], EXP['sV_exp_nmda'], EXP['Tv_exp_nmda'], EXP['Fout_all']],\
                                      YTICKS, YLIM, ['r','r','r','g']):
        ax.errorbar(np.linspace(-.2,.2,len(y1)), np.array(y1).mean(axis=1),\
                    yerr=np.array(y1).std(axis=1), marker='D', color='k')
        ax.errorbar(np.linspace(-.2,.2,len(x)), np.array(x).mean(axis=1),\
                    yerr=np.array(x).std(axis=1), marker='D', color=col)
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
for ax, ylabel in zip(AX[:,0], ['$\mu_V$ (mV)', '$\sigma_V$ (mV)', r'$\\tau_V$ (ms)', r'$\\nu$ (Hz)']):
    ax.set_ylabel(ylabel)
"""

if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser(description=
     """ 
     description of the whole modulus here
     """
    ,formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("--seed", type=int, help="seed fo random numbers",default=37)
    parser.add_argument("--SEED", type=int, help="number of changed SEEDS",default=3)
    parser.add_argument("--discret_sim", type=int, help="space discretization for numerical simulation", default=20)
    parser.add_argument("--tstop", type=float, help="max simulation time (s)", default=3000.)
    parser.add_argument("--dt", type=float, help="simulation time step (ms)", default=0.025)
    parser.add_argument("--discret_th", type=int, help="discretization for theoretical evaluation",default=20)
    parser.add_argument("--with_active_mechs", help="with active mechanisms for comparison", action="store_true")
    parser.add_argument("-u", "--update_plot", help="plot the figures", action="store_true")
    parser.add_argument( '-f', "--filename",help="filename",type=str, default='data.npz')

    args = parser.parse_args()

    if args.update_plot:
        data = dict(np.load(args.filename))
        data['plot'] = get_plotting_instructions()
        np.savez(args.filename, **data)
    else:
        from demo import *
        nrn.nrn_load_dll('../numerical_simulations/x86_64/.libs/libnrnmech.so')
        soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')
        stick['NSEG'] = args.discret_sim
    
        SET_OF_EXPS = create_set_of_exps(args)
    
        print(' first we set up the model [...]')
        x_exp, cables = setup_model(soma, stick, params)    
        ii=0
        ## MAKING THE BASELINE EXPERIMENT
        shtn_input = {'synchrony':synch_baseline,
                      'fe_prox':fe_baseline, 'fi_prox':fi_baseline,
                      'fe_dist':fe_baseline, 'fi_dist':fi_baseline}
        synchronous_stim={'location':[4,14], 'spikes':[]} # NONE !!!
        
        for s in range(args.SEED):
            print('baseline sim. , seed=', s)
            t, V, Vnmda, Vhh, Vall = run_all(args, shtn_input, cables, params, synchronous_stim, seed=(ii+args.seed+s*(s+2))**3%100)
            muV_soma, sV_soma, Tv_soma, Fhh, Fall = analyze_simulation(t, V, Vnmda, Vhh, Vall)
            
            for EXP in SET_OF_EXPS:
                EXP['muV_exp_ctrl'][int(N_POINTS/2.),s], EXP['sV_exp_ctrl'][int(N_POINTS/2.),s],\
                  EXP['Tv_exp_ctrl'][int(N_POINTS/2.),s] = muV_soma[0], sV_soma[0], Tv_soma[0]
                EXP['muV_exp_nmda'][int(N_POINTS/2.),s], EXP['sV_exp_nmda'][int(N_POINTS/2.),s],\
                  EXP['Tv_exp_nmda'][int(N_POINTS/2.),s] = muV_soma[1], sV_soma[1], Tv_soma[1]
                EXP['Fout_hh'][int(N_POINTS/2.),s], EXP['Fout_all'][int(N_POINTS/2.),s] = Fhh, Fall

        for EXP in SET_OF_EXPS:
            for i in np.delete(np.arange(N_POINTS), int(N_POINTS/2.)):
                shtn_input = {'synchrony':EXP['synchrony'][i],
                              'fe_prox':EXP['fe_prox'][i], 'fi_prox':EXP['fi_prox'][i],
                              'fe_dist':EXP['fe_dist'][i], 'fi_dist':EXP['fi_dist'][i]}
                for s in range(args.SEED):
                    print('sim=', i, ', seed=', s)
                    t, V, Vnmda, Vhh, Vall = run_all(args, shtn_input, cables, params, synchronous_stim, seed=(ii+args.seed+s*(s+1))**3%100)
                    muV_soma, sV_soma, Tv_soma, Fhh, Fall = analyze_simulation(t, V, Vnmda, Vhh, Vall)
                    EXP['muV_exp_ctrl'][i,s], EXP['sV_exp_ctrl'][i,s], EXP['Tv_exp_ctrl'][i,s] = muV_soma[0], sV_soma[0], Tv_soma[0]
                    EXP['muV_exp_nmda'][i,s], EXP['sV_exp_nmda'][i,s], EXP['Tv_exp_nmda'][i,s] = muV_soma[1], sV_soma[1], Tv_soma[1]
                    EXP['Fout_hh'][i,s], EXP['Fout_all'][i,s] = Fhh, Fall
                    
                ii+=1
                plt.close('all')
                
        np.savez(args.filename, args=args, plot=get_plotting_instructions())
        ii=0
        for EXP in SET_OF_EXPS:
            np.save(args.filename+str(ii)+'.npy', [EXP['muV_exp_ctrl'], EXP['sV_exp_ctrl'], EXP['Tv_exp_ctrl'],\
                                                   EXP['muV_exp_nmda'], EXP['sV_exp_nmda'], EXP['Tv_exp_nmda'], EXP['Fout_hh'], EXP['Fout_all']])
            ii+=1
        

