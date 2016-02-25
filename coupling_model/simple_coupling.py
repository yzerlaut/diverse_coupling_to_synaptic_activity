import numpy as np
import sys
sys.path.append('../')
from theory.analytical_calculus import get_the_fluct_prop_at_soma
from simple_synaptic_integration import get_fluct_var
from firing_response_description.template_and_fitting import final_func
from scipy.stats.stats import pearsonr

## LOADING ALL CELLS PROPERTIES
soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')
ALL_CELLS = np.load('../input_impedance_calibration/all_cell_params.npy')

def single_experiment(i_nrn, balance=-54e-3):
    
    ## FIRING RATE RESPONSE for control
    feG, fiG, feI, fiI, synch, muV, sV, TvN, muGn = get_fluct_var(i_nrn, balance=balance)
    Fout0 = final_func(ALL_CELLS[i_nrn]['P'], muV, sV, TvN,\
                      ALL_CELLS[i_nrn]['Gl'] , ALL_CELLS[i_nrn]['Cm'])

    fig, ax = plt.subplots(figsize=(4,3))
    
    ## Then for other protocols
    PROTOCOLS = ['unbalanced activity', 'proximal activity', 'distal activity', 'synchrony']
    coupling = np.zeros(len(PROTOCOLS)+2)
    coupling[0] = Fout0[0]
    coupling[1] = Fout0.mean()

    ax.plot(Fout0, 'k-', lw=2)
    COLORS=['r', 'b', 'g', 'c']

    ii=0
    for p, i  in zip(PROTOCOLS, range(2,len(coupling))):
        feG, fiG, feI, fiI, synch, muV, sV, TvN, muGn = get_fluct_var(i_nrn, exp_type=p, balance=balance)
        Fout2 = final_func(ALL_CELLS[i_nrn]['P'], muV, sV, TvN,\
                           ALL_CELLS[i_nrn]['Gl'], ALL_CELLS[i_nrn]['Cm'])
        coupling[i] = Fout2.mean()
        ax.plot(Fout2, '-', lw=2, color=COLORS[ii])
        ii+=1
        
    ax.set_title('cell'+str(i_nrn))
    set_plot(ax, xticks=[])
    fig.savefig('data/'+str(i_nrn)+'.svg')
    plt.close()
    
    return coupling


def correlating_electrophy_and_coupling(COUPLINGS, BIOPHYSICS, scale='lin'):
    """ plot of the correlation function"""

    ###############################################################################
    
    PROTOCOLS = ['unbalanced activity', 'proximal activity', 'distal activity', 'synchrony']
    
    YLABELS1 = [r"mean response level \\n $\langle \nu_\mathrm{out}\rangle$ (Hz)"]+\
               ['change of gain (Hz$^{-1}$) \n for increasing \n '+p for p in PROTOCOLS]
    
    E_LABELS = [r"$\langle V_\mathrm{thre}^\mathrm{eff} \rangle_\mathcal{D}$ (mV)",\
                r"$\langle \partial \nu / \partial \mu_V \rangle_\mathcal{D}$ (Hz/mV)",\
                r"$\langle \partial \nu / \partial \sigma_V \rangle_\mathcal{D}$ (Hz/mV)",\
                r"$\langle \partial \nu / \partial \tau_V^{N} \rangle_\mathcal{D}$ (Hz/%)"]

    X = [BIOPHYSICS[i,:] for i in range(BIOPHYSICS.shape[0])]
    Y = [COUPLINGS[i,:] for i in range(COUPLINGS.shape[0])]
    
    INDEXES, MARKER, SIZE = [21, 2, 27, 1], ['^', 'd', '*', 's'], [12, 11, 17, 10]

    fig, AX = plt.subplots(len(Y), len(X), figsize=(20,30))
    fig.subplots_adjust(wspace=.6, hspace=1.)
    for i in range(len(X)):
        for j in range(len(Y)):
            # for k in range(len(MARKER)):
            #     if scale=='log' and (Y[j][INDEXES[k]])>0:
            #         AX[j, i].plot([X[i][INDEXES[k]]], [np.log(Y[j][INDEXES[k]])/np.log(10)],\
            #                 lw=0, color='lightgray', marker=MARKER[k],
            #                 label='cell '+str(k), ms=SIZE[k])
            #     else:
            #         AX[j, i].plot([X[i][INDEXES[k]]], [Y[j][INDEXES[k]]],\
            #                 lw=0, color='lightgray', marker=MARKER[k],
            #                 label='cell '+str(k), ms=SIZE[k])
            
            cond = (Y[j]!=0) & (np.abs(Y[j])<1e4) & (np.abs(Y[j])>1e-2)
            xx, y = np.array(X[i][cond]), np.array(Y[j][cond])
            if scale=='log':
                y = np.log(y)/np.log(10)
                cc, pp = pearsonr(xx, y)
                x = np.linspace(xx.min(), xx.max())
                yy = np.polyval(np.polyfit(np.array(xx, dtype='f8'), np.array(y, dtype='f8'), 1), x)
            else:
                cc, pp = pearsonr(xx, y)
                x = np.linspace(xx.min(), xx.max())
                yy = np.polyval(np.polyfit(np.array(xx, dtype='f8'), np.array(y, dtype='f8'), 1), x)
            AX[j, i].plot(x, yy, 'k--', lw=.5)
            AX[j, i].plot(xx, y, 'ko')
            AX[j, i].annotate('c='+str(np.round(cc,1))+', p='+'%.1e' % pp,\
                         (0.15,1), xycoords='axes fraction')
            if i in [0,3]:
                AX[j, i].invert_xaxis()
            if scale=='log':
                set_plot(AX[j, i], ['left', 'bottom'],\
                         ylabel=YLABELS1[j],xlabel=E_LABELS[i],\
                         yticks=[-1,0,1,2],\
                         yticks_labels=['$10^{-1}$', '1', '$10^{1}$', '$10^{2}$'])
            else:
                set_plot(AX[j, i], ['left', 'bottom'], ylabel=YLABELS1[j], xlabel=E_LABELS[i])
    return fig


if __name__=='__main__':

    import matplotlib.pylab as plt
    sys.path.append('/home/yann/work/python_library/')
    from my_graph import set_plot, put_list_of_figs_to_svg_fig

    if sys.argv[-1]=='sim':
        COUPLINGS = np.zeros((6, len(ALL_CELLS)))
        BIOPHYSICS = np.zeros((4, len(ALL_CELLS)))
        for i_nrn in range(len(ALL_CELLS)):
            print 'cell', i_nrn
            BIOPHYSICS[:,i_nrn] = ALL_CELLS[i_nrn]['E']
            COUPLINGS[:,i_nrn] = single_experiment(i_nrn)
        np.save('data/elctrophy_vs_coupling.npy', [COUPLINGS, BIOPHYSICS])

    else:
        COUPLINGS, BIOPHYSICS = np.load('data/elctrophy_vs_coupling.npy')
        fig = correlating_electrophy_and_coupling(COUPLINGS, BIOPHYSICS,scale='log')
        fig.savefig('log_correlation.svg')
        fig = correlating_electrophy_and_coupling(COUPLINGS, BIOPHYSICS)
        fig.savefig('lin_correlation.svg')
    
        
    # plt.show()

