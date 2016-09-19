import numpy as np
import matplotlib.pylab as plt

from synaptic_integration import get_fluct_var
from firing_responses import single_experiment
from scipy.stats.stats import pearsonr

import sys
sys.path.append('/home/yann/work/python_library')
from my_graph import set_plot

ALL_CELLS = np.load('../ball_and_rall_tree/all_cell_params.npy')

from firing_response_description.template_and_fitting import final_func

def rectify(x):
    """ take an array and replaces its negative points and replace it with 0 values """
    out = x.copy()
    out[out<0]=0
    return out

def generate_population_rate(t, F0=200., SF=100., T=0.04, seed=1):
    """ TO BE DONE, PUT INSIDE THIS THE SPATIO TEMPORAL PROP OF THE POP RATES !!! """
    np.random.seed(seed)
    WN = np.random.randn(len(t)) # white noise 1
    F = 1./(1.+((t-t.mean())/T)**2) # filter
    CN =np.convolve(WN, F-F.mean(), mode='same')*(t[1]-t[0]) # colored noise
    return rectify(F0+SF/CN.std()*CN)

def run_single_experiment(t, i_nrn, args, exp_type='control', seed=1, F_discretization=100):
    """
    run the population stimulation protocol,
    for different paradigm :
    - control : mixed of proximal and distal input with a given statistics
    - unbalanced : a small bias toward excitation make that the mean is not constant anymore
    - correlated : same mean but more variance for the presynaptic rates
    - proximal : only proximal stimulation, no distal stimulation
    - distal : only distal stimulation, no proximal stimulation
    """

    # setting up the experiment
    F = generate_population_rate(t, F0=args.F0, SF=args.SF, T=args.TF, seed=seed) # basis

    ### We discretize the problem to avoid too long simulations
    F_discret = np.linspace(F.min(), F.max(), F_discretization)
    feG_discret, fiG_discret, feI_discret, fiI_discret, synch_discret,\
      muV_discret, sV_discret, TvN_discret, muGn_discret, Fout_discret = [0*F_discret for i in range(10)]

    feG_discret, fiG_discret, feI_discret, fiI_discret, synch_discret,\
      muV_discret, sV_discret, TvN_discret, muGn_discret =\
      get_fluct_var(i_nrn, F_discret, exp_type=exp_type)
    Fout_discret = final_func(ALL_CELLS[i_nrn]['P'], muV_discret, sV_discret, TvN_discret,\
                  ALL_CELLS[i_nrn]['Gl'], ALL_CELLS[i_nrn]['Cm'])
    
    feG, fiG, feI, fiI, synch, muV, sV, TvN, muGn, Fout = [0*F for i in range(10)]
    
    for i in range(len(F)):
        i0 = np.argmin((F_discret-F[i])**2)
        feG[i], fiG[i], feI[i], fiI[i], synch[i],\
          muV[i], sV[i], TvN[i], muGn[i], Fout[i] =\
           feG_discret[i0], fiG_discret[i0], feI_discret[i0], fiI_discret[i0], synch_discret[i0],\
           muV_discret[i0], sV_discret[i0], TvN_discret[i0], muGn_discret[i0], Fout_discret[i0]
         
    return feG, fiG, feI, fiI, muV, sV, TvN, muGn, Fout

        
def make_fig(args):
            
    NEURONS, PROTOCOLS = list(args.NEURONS), args.PROTOCOLS
    COLORS, SEEDS = args.COLORS, args.SEEDS
    
    t = np.arange(int(args.tstop/args.dt)-1)*args.dt # time array
                              
    fig1, AX1 = plt.subplots(4, len(PROTOCOLS), figsize=(4.*len(PROTOCOLS),10))
    fig2, AX2 = plt.subplots(4, len(PROTOCOLS), figsize=(4.*len(PROTOCOLS),10))
    fig3, AX3 = plt.subplots(len(NEURONS), len(PROTOCOLS), figsize=(4.*len(PROTOCOLS), 3*len(NEURONS)))

    [FEG, FIG, FEI, FII, MUV, SV, TVN, MUGN] = [np.zeros((len(PROTOCOLS), 2, len(t))) for i in range(8)]
    FOUT = np.zeros((len(NEURONS), len(PROTOCOLS), 2, len(t)))

    for ip in range(len(PROTOCOLS)):
        for i_nrn in range(len(NEURONS)):
            FEG[ip, 0, :], FIG[ip, 0, :], FEI[ip, 0, :], FII[ip, 0, :],\
               MUV[ip, 0, :], SV[ip, 0, :], TVN[ip, 0, :], MUGN[ip, 0, :],\
               FOUT[i_nrn, ip, 0, :] = \
               run_single_experiment(t, int(NEURONS[i_nrn]),\
                        args, exp_type='non specific activity', seed=SEEDS[ip])
            FEG[ip, 1, :], FIG[ip, 1, :], FEI[ip, 1, :], FII[ip, 1, :],\
               MUV[ip, 1, :], SV[ip, 1, :], TVN[ip, 1, :], MUGN[ip, 1, :],\
               FOUT[i_nrn, ip, 1, :] = \
               run_single_experiment(t, int(NEURONS[i_nrn]),\
                        args, exp_type=PROTOCOLS[ip], seed=SEEDS[ip])
    ### PLOTTING ALL
    for ip, color in zip(list(range(len(PROTOCOLS))), COLORS):
        # membrane potential quantities
        for j, x in zip(list(range(4)), [1e3*MUV[ip,1,:], 1e3*SV[ip,1,:],\
                    1e2*TVN[ip,1,:], MUGN[ip,1,:]]):
            AX1[j, ip].plot(t, x, color=color, lw=3, label=PROTOCOLS[ip])
        for j, x in zip(list(range(4)), [1e3*MUV[ip,0,:], 1e3*SV[ip,0,:],\
                    1e2*TVN[ip,0,:], MUGN[ip,0,:]]):
            AX1[j, ip].plot(t, x, 'k-', lw=1, label='non specific activity')
        # presynaptic activity
        for j, x in zip(list(range(4)), [FEG[ip,1,:], FIG[ip,1,:], FEI[ip,1,:], FII[ip,1,:]]):
            AX2[j, ip].plot(t, x, color=color, lw=3, label=PROTOCOLS[ip])
        for j, x in zip(list(range(4)), [FEG[ip,0,:], FIG[ip,0,:], FEI[ip,0,:], FII[ip,0,:]]):
            AX2[j, ip].plot(t, x, 'k-', lw=1, label='non specific activity')
        # then output firing
        for i_nrn in range(len(NEURONS)):
            AX3[i_nrn, ip].plot(t, FOUT[i_nrn, ip, 0, :], 'k-', lw=1, label='non specific activity')
            AX3[i_nrn, ip].plot(t, FOUT[i_nrn, ip, 1, :], color=color, lw=3, label=PROTOCOLS[ip])
            
    # THEN LIMITS
    for ip in range(len(PROTOCOLS)):
        for j, x in zip(list(range(4)), [FEG, FIG, FEI, FII]):
            AX2[j, ip].plot([0,0], [x.min(),x.max()], color='w', alpha=0)
        for j, x in zip(list(range(4)), [1e3*MUV, 1e3*SV,\
                    1e2*TVN, MUGN]):
            AX1[j, ip].plot([0,0], [x.min(),x.max()], color='w', alpha=0)

    # then labels
    YLABELS1 = ['$\mu_V$ (mV)', '$\sigma$ (mV)', '$\\tau_V / \\tau_m^0$ (%)', '$\mu_G / g_L$']
    YLABELS2 = [r'$\nu_e^\mathrm{prox}$ (Hz)', r'$\nu_i^\mathrm{prox}$ (Hz)',\
                   r'$\nu_e^\mathrm{dist}$ (Hz)', r'$\nu_i^\mathrm{dist}$ (Hz)']
    for AX, YLABELS in zip([AX1, AX2], [YLABELS1, YLABELS2]):
        for i in range(AX.shape[0]):
            for j in range(AX.shape[1]):
                if i==0:
                    AX[i,j].legend(bbox_to_anchor=(.8, 1.4), prop={'size':'small'})
                if j==0 and i==AX.shape[1]-1:
                    set_plot(AX[i,j], ylabel=YLABELS[i], xlabel='time (s)')
                elif j==0:
                    set_plot(AX[i,j], ylabel=YLABELS[i], xlabel='time (s)')
                elif i==AX.shape[0]-1:
                    set_plot(AX[i,j], xlabel='time (s)')
                else:
                    set_plot(AX[i,j])
    NEURONS[:4] = ['cell1', 'cell2', 'cell3', 'cell4'] # renaming cells !
    for i in range(AX3.shape[0]):
        for j in range(AX3.shape[1]):
            if i==0:
                AX3[i,j].legend(bbox_to_anchor=(.8, 1.4), prop={'size':'small'})
            if j==0 and i==AX3.shape[1]-1:
                set_plot(AX3[i,j], ylabel=r'$\nu_\mathrm{out}$ (Hz)', xlabel='time (s)')
                AX3[i,j].annotate(NEURONS[i], (-.4, .5), xycoords='axes fraction')
            elif j==0:
                set_plot(AX3[i,j], ylabel=r'$\nu_\mathrm{out}$ (Hz)')
                AX3[i,j].annotate(NEURONS[i], (-.4, .5), xycoords='axes fraction')
            elif i==AX3.shape[0]-1:
                set_plot(AX3[i,j], xlabel='time (s)')
            else:
                set_plot(AX3[i,j])

    return fig1, fig2, fig3

def CC_func(X, Y, steps):
    X -= X.mean()
    # Y = (Y-Y.mean())/Y.std()
    CC = np.correlate(X[steps:],Y)/steps
    return CC
    
def get_cross_correlation_functions(args):
            
    NEURONS, PROTOCOLS = list(args.NEURONS), args.PROTOCOLS
    COLORS, SEEDS = args.COLORS, args.SEEDS
    print(args.NEURONS)
    
    t = np.arange(int(args.tstop*args.time_increase_factor/args.dt)-1)*args.dt # time array
    # now for cross correlation analysis
    steps = int(args.window_for_cross_correl/(t[1]-t[0])) # number of steps to sum on
    t_shift = np.arange(steps+1)*(t[1]-t[0])
                              
    fig, AX = plt.subplots(len(NEURONS), 1, figsize=(4, 3*len(NEURONS)))
    plt.subplots_adjust(left=.2)
    fig.suptitle('coupling with\n presynaptic population \n (stPR)')

    [FEG, FIG, FEI, FII, MUV, SV, TVN, MUGN] = [np.zeros((len(PROTOCOLS), 2, len(t))) for i in range(8)]
    FOUT = np.zeros((len(NEURONS), len(PROTOCOLS), 2, len(t)))

    for i_nrn in range(len(NEURONS)):
        print('cell', i_nrn)
        for ip in range(len(PROTOCOLS)):
            print('-- protocol: ', PROTOCOLS[ip])
            feG, _, feI, _, _, _, _, _, Fout = \
               run_single_experiment(t, i_nrn, args, exp_type=PROTOCOLS[ip])
            X, Y = 1./2.*(feG+feI), Fout*(t[1]-t[0])
            CC = CC_func(X, Y, steps)
            AX[i_nrn].plot(t_shift, CC, color=COLORS[ip], lw=2)
            AX[i_nrn].plot(-t_shift, CC, color=COLORS[ip], lw=2)
        if i_nrn==len(NEURONS)-1:
            set_plot(AX[i_nrn], ylabel='Hz', xlabel='time shift (s)')
        else:
            set_plot(AX[i_nrn], ylabel='Hz')

    return fig

def coupling_over_data(args, params):
    # loading data
    INDEX, FOUT, SFOUT, MUV, SV, TVN, MUGN, GL, CM, EL,\
        V0, DmuV, DsV, DTv, E0, EmuV, EsV, ETv = \
            np.load('../3d_scan/data/full_data.npy')
    
    PROTOCOLS = args.PROTOCOLS
    COUPLINGS = np.zeros((len(INDEX), len(PROTOCOLS)+1))
    
    t = np.arange(int(args.tstop*args.time_increase_factor/args.dt)-1)*args.dt # time array
    steps = int(args.window_for_cross_correl/(t[1]-t[0])) # number of steps to sum on
    
    for i_nrn in range(len(INDEX)):
        feG, _, feI, _, _, _, _, _, Fout = \
               run_single_experiment(t, INDEX[i_nrn],\
                        args, params, exp_type='non specific activity')
        X, Y = 1./2.*(feG+feI), Fout*(t[1]-t[0])
        X -= X.mean() 
        CC0 = np.correlate(X[steps:],Y)[0]/steps # non specific stimulation
        COUPLINGS[i_nrn,0]  = Fout.mean()
        COUPLINGS[i_nrn,1]  = CC0
        for ip in range(1, len(PROTOCOLS)):
            feG, _, feI, _, _, _, _, _, Fout = \
               run_single_experiment(t, INDEX[i_nrn],\
                        args, params, exp_type=PROTOCOLS[ip])
            X, Y = 1./2.*(feG+feI), Fout*(t[1]-t[0])
            # in percent
            COUPLINGS[i_nrn,ip+1]  = 1e2*(CC_func(X, Y, steps)[0]-CC0)/CC0
    return COUPLINGS


def removing_cells(INDEX2, E02, EmuV2, EsV2, ETv2, COUPLINGS2, cells=['cell31']):
    INDEX, E0, EmuV, EsV, ETv = [], [], [], [], []
    PROTOCOLS, j = args.PROTOCOLS, 0
    COUPLINGS = np.zeros((len(INDEX2)-len(cells), len(PROTOCOLS)+1))
    for i in range(len(INDEX2)):
        if INDEX2[i] not in cells:
            for x in ['INDEX', 'E0', 'EmuV', 'EsV', 'ETv']:
                exec(x+'.append('+x+'2[i])')
            COUPLINGS[j,:] = COUPLINGS2[i,:]
            j+=1
    return np.array(INDEX),np.array(E0), np.array(EmuV), np.array(EsV), np.array(ETv), COUPLINGS
    

def histogram_of_couplings(args, params, bins=10):
    """ plot """

    INDEX, _, _, _, _, _, _, _, _, _, _, _, _, _,\
        E0, EmuV, EsV, ETv = \
            np.load('../3d_scan/data/full_data.npy')
    COUPLINGS = coupling_over_data(args, params)

    ###############################################################################
    """ removing cell31 !!!!!! """
    INDEX, E0, EmuV, EsV, ETv, COUPLINGS = removing_cells(INDEX, E0, EmuV,\
                          EsV, ETv, COUPLINGS, cells=['cell31'])
    ###############################################################################
    YLABELS1 = [r"$\langle \nu_\mathrm{out}\rangle$"]+\
               [r"coupling (Hz)"]+\
               ['coupling increase (%) \n for '+p for p in args.PROTOCOLS[1:]]
    
    fig, AX = plt.subplots(1, len(args.PROTOCOLS)+1, figsize=(3*(1+len(args.PROTOCOLS)),2.5))
    fig.subplots_adjust(bottom=.3,left=.1,wspace=.3, hspace=.3)

    for ax, i, label in zip(AX, list(range(len(args.PROTOCOLS)+1)), YLABELS1):
        hist, be = np.histogram(COUPLINGS[:,i], bins=bins)
        ax.bar(.5*(be[:-1]+be[1:]), hist, width=be[1]-be[0], color='k', alpha=.5)
        set_plot(ax, xlabel=label, ylabel='cell #')
    return fig

def correlating_electrophy_and_coupling(args, params):
    """ plot """

    INDEX, _, _, _, _, _, _, _, _, _, _, _, _, _,\
        E0, EmuV, EsV, ETv = \
            np.load('../3d_scan/data/full_data.npy')
    COUPLINGS = coupling_over_data(args, params)

    # ###############################################################################
    # """ removing cell31 !!!!!! """
    # INDEX, E0, EmuV, EsV, ETv, COUPLINGS = removing_cells(INDEX, E0, EmuV,\
    #                                                       EsV, ETv, COUPLINGS, cells=['cell31'])
    # ###############################################################################
    
    YLABELS1 = [r"$\langle \nu_\mathrm{out}\rangle$"]+\
               [r"coupling (Hz)"]+\
               ['coupling increase (%) \n for '+p for p in args.PROTOCOLS[1:]]
    
    E_LABELS = [r"$\langle V_\mathrm{thre}^\mathrm{eff} \rangle_\mathcal{D}$ (mV)",\
                r"$\langle \partial \nu / \partial \mu_V \rangle_\mathcal{D}$ (Hz/mV)",\
                r"$\langle \partial \nu / \partial \sigma_V \rangle_\mathcal{D}$ (Hz/mV)",\
                r"$\langle \partial \nu / \partial \tau_V^{N} \rangle_\mathcal{D}$ (Hz/%)"]

    X = [E0, EmuV, EsV, ETv]
    Y = [COUPLINGS[:,i] for i in range(COUPLINGS.shape[1])]

    INDEXES, MARKER, SIZE = [], ['^', 'd', '*', 's'], [12, 11, 17, 10]
    for cell in args.NEURONS:
        print(cell, np.where(np.array(INDEX)==cell))
        INDEXES.append(np.where(INDEX==cell)[0][0])

    fig, AX = plt.subplots(len(Y), len(X), figsize=(20,18))
    fig.subplots_adjust(wspace=.3, hspace=.3)
    for i in range(len(X)):
        for j in range(len(Y)):
            for k in range(len(MARKER)):
                AX[j, i].plot([X[i][INDEXES[k]]], [Y[j][INDEXES[k]]],\
                        lw=0, color='lightgray', marker=MARKER[k],
                        label='cell '+str(k+1), ms=SIZE[k])
                AX[j, i].plot([X[i][INDEXES[k]]], [Y[j][INDEXES[k]]],\
                        lw=0, color='k', marker='o')
            AX[j, i].scatter(X[i], Y[j], color='k', marker='o')
            cc, pp = pearsonr(X[i], Y[j])
            x = np.linspace(X[i].min(), X[i].max())
            AX[j, i].plot(x, np.polyval(np.polyfit(np.array(X[i], dtype='f8'), np.array(Y[j], dtype='f8'), 1), x), 'k--', lw=.5)
            AX[j, i].annotate('c='+str(np.round(cc,1))+', p='+'%.1e' % pp,\
                         (0.15,1), xycoords='axes fraction')
            if i in [0,3]:
                AX[j, i].invert_xaxis()
            set_plot(AX[j, i], ['left', 'bottom'], ylabel=YLABELS1[j], xlabel=E_LABELS[i])
    return fig
        


import argparse

if __name__=='__main__':

    parser=argparse.ArgumentParser(description=
     """ 
     population coupling protocols
     """
    ,formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("--NEURONS", default=['1', '2', '3', '4'],\
                        help="Choose a neuron (model or neuronal ID)", nargs='*')
    # parser.add_argument("--PROTOCOLS",\
    #     default= ['non specific activity', 'unbalanced activity', 'proximal activity', 'distal activity'],
    #     help="Different paradigm tested")
    parser.add_argument("--PROTOCOLS",\
        default= ['non specific activity', 'unbalanced activity', 'proximal activity', 'distal activity', 'synchrony'],
        help="Different paradigm tested")
    parser.add_argument("--COLORS", default=['k', 'r', 'b', 'g', 'c'],\
                        help="Colors for protocols")
    parser.add_argument("--SEEDS", default=[32, 3, 2, 1, 13, 4, 8],\
                        help="Seeds for protocols")
    
    parser.add_argument("--F0",type=float, default=.3, help="mean input exc. frequency (Hz)")
    parser.add_argument("--SF",type=float, default=.15, help="std dev. input exc. frequency (Hz)")
    parser.add_argument("--TF",type=float, default=0.04, help="correlation time of input exc. frequency (s)")
    parser.add_argument("--factor_for_unbalancing", type=float, default=.2,\
                        help="additional percentage of excitatory input that creates the break of balance")
    parser.add_argument("--factor_for_prox", type=float, default=2.,\
                        help="factor to multiply the input, when only the proximal pop.")
    parser.add_argument("--factor_for_dist", type=float, default=2.,\
                       help="factor to multiply the input, when only the distal pop.")
    parser.add_argument("--fraction_distal_over_prox_for_distal", type=float, default=.95,\
                        help="fraction_distal_over_prox_for_distal")
    parser.add_argument("--sharpening_for_correlation", type=float, default=.6,\
                        help="sharpening window to put in mexican hat when CORRELATED (s)")
    parser.add_argument("--factor_for_correlation", type=float, default=1.5,\
                        help="sharpening window to put in mexican hat when CORRELATED (s)")

    parser.add_argument("--dt",type=float, default=5e-3, help="time step (s)")
    parser.add_argument("--tstop",type=float, default=2., help="total duration (s)")
    
    parser.add_argument("--COUPLING", help="get the cross correlations and coupling", action="store_true")
    parser.add_argument("--time_increase_factor",type=float, default=10., help="need a long time for cross correlations (s)")
    parser.add_argument("--window_for_cross_correl",type=float, default=.5, help="window for cross correl plot (s)")
    
    parser.add_argument("--HISTOGRAMS", help="histograms of couplings", action="store_true")
    parser.add_argument("--bins_for_histograms", help="histograms bin number", type=int, default=10)
    parser.add_argument("--CORRELATIONS", help="correlations between coupling and electrophy", action="store_true")
    
    parser.add_argument("--ALL", help="all figs and save them", action="store_true")
    args = parser.parse_args()

    if args.COUPLING:
        fig = get_cross_correlation_functions(args)
        plt.show()
    elif args.CORRELATIONS:
        fig = correlating_electrophy_and_coupling(args)
        plt.show()
    elif args.HISTOGRAMS:
        fig = histogram_of_couplings(args, bins=args.bins_for_histograms)
        plt.show()
    elif args.ALL:
        fig1, fig2, fig3 = make_fig(args)
        fig4 = get_cross_correlation_functions(args)
        fig5 = correlating_electrophy_and_coupling(args)
        put_list_of_figs_to_svg_fig([fig1, fig2, fig3, fig4, fig5], fig_name='fig.svg')
    else:
        fig1, fig2, fig3 = make_fig(args)
        plt.show()

