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
from firing_response_description.template_and_fitting import final_func
from demo import *

N_POINTS = 5
N_POINTS_TH = 10
inh_factor = 5.8

fe_baseline, fi_baseline, synch_baseline = 0.2, 0.2*inh_factor, 0.05
fe_vector = np.array([0.,0.1,fe_baseline,0.3,0.4])
fi_vector = np.round(fe_vector*inh_factor,1)
fe_vector_th = np.linspace(0., 0.4, N_POINTS_TH)
fi_vector_th = fe_vector_th*inh_factor
synch_vector = np.array([0.0, 0.02, synch_baseline, 0.15, 0.4])
synch_vector_plot = np.array([0.01, 0.02, synch_baseline, 0.15, 0.4])
synch_vector_th_plot=np.logspace(np.log(0.01)/np.log(10),np.log(0.4)/np.log(10),N_POINTS_TH)
synch_vector_th = synch_vector_th_plot
synch_vector_th[0] = 0

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
                    'xticks':synch_vector_plot,
                    'fe_prox':np.ones(N_POINTS)*fe_baseline,
                    'fe_dist':np.ones(N_POINTS)*fe_baseline,
                    'fi_prox':np.ones(N_POINTS)*fi_baseline,
                    'fi_dist':np.ones(N_POINTS)*fi_baseline,
                    'synchrony':synch_vector,
                    'muV_exp':np.zeros((N_POINTS, args.SEED)), 'sV_exp':np.zeros((N_POINTS, args.SEED)), 'Tv_exp':np.zeros((N_POINTS, args.SEED))
                    }]
    return SET_OF_EXPS

def create_set_of_exps_th(args):
    SET_OF_EXPS = [\
                   {'label':'$\\nu_e^p$ (Hz) \n prox. exc. ',
                    'xticks':fe_vector_th,
                    'fe_prox':fe_vector_th,
                    'fe_dist':np.ones(N_POINTS_TH)*fe_baseline,
                    'fi_prox':np.ones(N_POINTS_TH)*fi_baseline,
                    'fi_dist':np.ones(N_POINTS_TH)*fi_baseline,
                    'synchrony':np.ones(N_POINTS_TH)*synch_baseline,
                    'muV':np.zeros(N_POINTS_TH), 'sV':np.zeros(N_POINTS_TH), 'Tv':np.zeros(N_POINTS_TH)
                    },
                   {'label':'$\\nu_i^p$ (Hz) \n prox. inh. ',
                    'xticks':fi_vector_th,
                    'fe_prox':np.ones(N_POINTS_TH)*fe_baseline,
                    'fe_dist':np.ones(N_POINTS_TH)*fe_baseline,
                    'fi_prox':fi_vector_th,
                    'fi_dist':np.ones(N_POINTS_TH)*fi_baseline,
                    'synchrony':np.ones(N_POINTS_TH)*synch_baseline,
                    'muV':np.zeros(N_POINTS_TH), 'sV':np.zeros(N_POINTS_TH), 'Tv':np.zeros(N_POINTS_TH)
                    },
                   {'label':'$\\nu_e^d$ (Hz) \n distal exc. ',
                    'xticks':fe_vector_th,
                    'fe_prox':np.ones(N_POINTS_TH)*fe_baseline,
                    'fe_dist':fe_vector_th,
                    'fi_prox':np.ones(N_POINTS_TH)*fi_baseline,
                    'fi_dist':np.ones(N_POINTS_TH)*fi_baseline,
                    'synchrony':np.ones(N_POINTS_TH)*synch_baseline,
                    'muV':np.zeros(N_POINTS_TH), 'sV':np.zeros(N_POINTS_TH), 'Tv':np.zeros(N_POINTS_TH)
                    },
                   {'label':'$\\nu_i^d$ (Hz) \n distal inh.',
                    'xticks':fi_vector_th,
                    'fe_prox':np.ones(N_POINTS_TH)*fe_baseline,
                    'fe_dist':np.ones(N_POINTS_TH)*fe_baseline,
                    'fi_prox':np.ones(N_POINTS_TH)*fi_baseline,
                    'fi_dist':fi_vector_th,
                    'synchrony':np.ones(N_POINTS_TH)*synch_baseline,
                    'muV':np.zeros(N_POINTS_TH), 'sV':np.zeros(N_POINTS_TH), 'Tv':np.zeros(N_POINTS_TH)
                    },
                   {'label':'synchrony',
                    'xlabel':'synchrony',
                    'xticks':synch_vector_th_plot,
                    'fe_prox':np.ones(N_POINTS_TH)*fe_baseline,
                    'fe_dist':np.ones(N_POINTS_TH)*fe_baseline,
                    'fi_prox':np.ones(N_POINTS_TH)*fi_baseline,
                    'fi_dist':np.ones(N_POINTS_TH)*fi_baseline,
                    'synchrony':synch_vector_th,
                    'muV':np.zeros(N_POINTS_TH), 'sV':np.zeros(N_POINTS_TH), 'Tv':np.zeros(N_POINTS_TH)
                    }]
    return SET_OF_EXPS


def get_plotting_instructions():
    return """
args = data['args'].all()
from final_check_accuracy import *
SET_OF_EXPS = create_set_of_exps(args)
SET_OF_EXPS_TH = create_set_of_exps_th(args)
ii=0
for EXP in SET_OF_EXPS:
    EXP['muV_exp'], EXP['sV_exp'], EXP['Tv_exp'] = np.load(args.filename+str(ii)+'.npy')
    ii+=1
ii=0
for EXP in SET_OF_EXPS_TH:
    EXP['muV'],EXP['sV'],EXP['Tv']=np.load(args.filename+str(ii)+'_th.npy')
    ii+=1
sys.path.append('../../')
from common_libraries.graphs.my_graph import set_plot
fig, AX = plt.subplots(3, 5, figsize=(10,8))
plt.subplots_adjust(left=.25, bottom=.25, wspace=.4, hspace=.4)
# plotting all points in all plots so that they have the same boundaries !!
YTICKS = [[-70,-60,-50], [3,5,7], [12, 20, 28], [0,15,30]]
YLIM = [[-75,-30], [1.9,10.], [9,45], [-2,40]]
SV_MINIM = np.load('data/data_minim.npy')
for EXP, EXP_TH, sv_min, ii in zip(SET_OF_EXPS, SET_OF_EXPS_TH, SV_MINIM, list(range(len(SET_OF_EXPS)))):
    X = EXP['xticks']
    xticks = EXP['xticks'][::2]
    XTH = EXP_TH['xticks']
    XLIM = [X[0]-(X[1]-X[0])/3., X[-1]+(X[1]-X[0])/3.]
    for ax, y1, y2, yticks, ylim in zip(\
            AX[:,ii],\
            [EXP['muV_exp'], EXP['sV_exp'], EXP['Tv_exp']],\
            [EXP_TH['muV'], EXP_TH['sV'], EXP_TH['Tv']],\
            YTICKS, YLIM):
        if (ax in AX[1,:]):
            ax.plot(XTH, sv_min, '--', color='gray', lw=3)
        ax.plot(np.ones(2)*X[1], ylim, 'wD', lw=0, alpha=0., ms=0.01)
        ax.plot(XTH, y2, '-', color='gray', lw=3)
        ax.errorbar(X, np.array(y1).mean(axis=1),\
                    yerr=np.array(y1).std(axis=1), color='k', fmt='o', mfc='white')
        if (ax in AX[:,-1]):
            ax.set_xscale("log")
            XLIM = [0.005,0.5]
            xticks = np.concatenate([np.arange(10)*0.01+0.01,np.arange(5)*0.1+0.1])
        if (ax==AX[-1,0]):
            set_plot(ax, xlim=XLIM,\
                     xlabel=EXP['label'], xticks=xticks, yticks=yticks, ylim=ylim)
        elif (ax==AX[-1,ii]):
            set_plot(ax, xlim=XLIM, yticks_labels=[], xticks=xticks,
                     xlabel=EXP['label'], yticks=yticks, ylim=ylim)
        elif ax in AX[:,0]:
            set_plot(ax, xlim=XLIM, xticks=xticks,\
                     xticks_labels=[], yticks=yticks, ylim=ylim)
        else:
            set_plot(ax, xlim=XLIM, yticks_labels=[], xticks_labels=[],\
                    yticks=yticks, ylim=ylim, xticks=xticks)
for ax, ylabel in zip(AX[:,0], ['$\mu_V$ (mV)', '$\sigma_V$ (mV)', r'$\\tau_V$ (ms)']):
    ax.set_ylabel(ylabel)
"""

if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser(description=
     """ 
     description of the whole modulus here
     """
    ,formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("--SIM", action='store_true') # flag for running simuluation !
    parser.add_argument("--find_synchrony_min", action='store_true') # find synchrony min
    parser.add_argument("--seed", type=int, help="seed fo random numbers",default=37)
    parser.add_argument("--SEED", type=int, help="number of changed SEEDS",default=3)
    parser.add_argument("--discret_sim", type=int, help="space discretization for numerical simulation", default=20)
    parser.add_argument("--tstop", type=float, help="max simulation time (s)", default=3000.)
    parser.add_argument("--dt", type=float, help="simulation time step (ms)", default=0.025)
    parser.add_argument("--discret_th", type=int, help="discretization for theoretical evaluation",default=20)
    
    parser.add_argument("-u", "--update_plot", help="plot the figures", action="store_true")
    parser.add_argument( '-f', "--filename",help="filename",type=str, default='data/data.npz')

    args = parser.parse_args()

    SET_OF_EXPS = create_set_of_exps(args)
    
    soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')
    x_exp, cables = setup_model(soma, stick, params)
    
    if args.update_plot:
        data = dict(np.load(args.filename))
        data['plot'] = get_plotting_instructions()
        np.savez(args.filename, **data)
    elif args.SIM:
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
                                                      
                ii+=1
                plt.close('all')
        np.savez(args.filename, args=args, plot=get_plotting_instructions())
        ii=0
        for EXP in SET_OF_EXPS:
            np.save(args.filename+str(ii)+'.npy', [EXP['muV_exp'], EXP['sV_exp'], EXP['Tv_exp']])
            ii+=1

    elif args.find_synchrony_min:
        from scipy.optimize import minimize # MINIMIZATION
        # loading min data
        SET_OF_EXPS = create_set_of_exps(args)
        SV_EXP = []
        ii=0
        for EXP in SET_OF_EXPS:
            _, sV_exp, _ = np.load(args.filename+str(ii)+'.npy')
            SV_EXP.append(sV_exp.mean(axis=1))
        # function to minimize
        def to_minimize(synch):
            SV_TH = []
            for EXP in SET_OF_EXPS:
                SHTN_INPUT =  {'synchrony':EXP['synchrony']+synch,
                               'fe_prox':EXP['fe_prox'], 'fi_prox':EXP['fi_prox'],
                               'fe_dist':EXP['fe_dist'], 'fi_dist':EXP['fi_dist']}
                _, sV, _, _ = get_the_fluct_prop_at_soma(SHTN_INPUT, params, soma, stick)
                SV_TH.append(1e3*sV)
                print(1e3*sV)

            print(synch,np.array([(sexp-sth)**2 for sexp, sth in zip(SV_EXP, SV_TH)]).sum())
            return np.array([(sexp-sth)**2 for sexp, sth in zip(SV_EXP, SV_TH)]).sum()
        res = minimize(to_minimize, 0.2, options={'maxiter': 5})
        print(res)
        SET_OF_EXPS_TH = create_set_of_exps_th(args)
        SV_TH = []
        for EXP in SET_OF_EXPS_TH:
            SHTN_INPUT =  {'synchrony':EXP['synchrony']+res.x[0],
                           'fe_prox':EXP['fe_prox'], 'fi_prox':EXP['fi_prox'],
                           'fe_dist':EXP['fe_dist'], 'fi_dist':EXP['fi_dist']}
            _, sV, _, _ = get_the_fluct_prop_at_soma(SHTN_INPUT, params, soma, stick)
            SV_TH.append(1e3*sV)
        # then save the copy to be plotted
        np.save('data/data_minim.npy', SV_TH)
            
    else: # theoretical evaluation
        SET_OF_EXPS_TH = create_set_of_exps_th(args)
        Tm0 = get_membrane_time_constants(soma, stick, params)
        ii=0
        for EXP in SET_OF_EXPS_TH:
            SHTN_INPUT =  {'synchrony':EXP['synchrony'],
                            'fe_prox':EXP['fe_prox'], 'fi_prox':EXP['fi_prox'],
                            'fe_dist':EXP['fe_dist'], 'fi_dist':EXP['fi_dist']}
            print(SHTN_INPUT)
            muV, sV, Tv, _ = get_the_fluct_prop_at_soma(SHTN_INPUT, params, soma, stick)
            print(1e3*muV, 1e3*sV, 1e3*Tm0*Tv)
            np.save(args.filename+str(ii)+'_th.npy', [1e3*muV, 1e3*sV, 1e3*Tm0*Tv])
            ii+=1

        
