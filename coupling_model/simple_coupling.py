import numpy as np
import sys
sys.path.append('../')
from theory.analytical_calculus import get_the_fluct_prop_at_soma
from synaptic_integration import get_fluct_var
from firing_response_description.template_and_fitting import final_func
from scipy.stats.stats import pearsonr

## LOADING ALL CELLS PROPERTIES
soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')
ALL_CELLS = np.load('../input_impedance_calibration/all_cell_params.npy')

def single_experiment(F, i_nrn):
    
    ## FIRING RATE RESPONSE for control
    feG, fiG, feI, fiI, synch, muV, sV, TvN, muGn = get_fluct_var(i_nrn, F)
    Fout0 = final_func(ALL_CELLS[i_nrn]['P'], muV, sV, TvN,\
                      ALL_CELLS[i_nrn]['Gl'], ALL_CELLS[i_nrn]['Cm'])

    ## Then for other protocols
    
    PROTOCOLS = ['unbalanced activity', 'proximal activity', 'distal activity', 'synchrony']
    coupling = np.zeros(len(PROTOCOLS)+2)
    coupling[0] = Fout0.mean()
    coupling[1] = np.diff(Fout0).mean()/np.diff(F).mean()
    
    for p, i  in zip(PROTOCOLS, range(1,len(coupling))):
        feG, fiG, feI, fiI, synch, muV, sV, TvN, muGn = get_fluct_var(i_nrn, F, exp_type=p)
        Fout = final_func(ALL_CELLS[i_nrn]['P'], muV, sV, TvN,\
                          ALL_CELLS[i_nrn]['Gl'], ALL_CELLS[i_nrn]['Cm'])
        if np.diff(Fout0).mean()>0:
            coupling[i] = (np.diff(Fout).mean()-np.diff(Fout0).mean())/np.diff(Fout0).mean()
    return coupling

def correlating_electrophy_and_coupling(COUPLINGS, BIOPHYSICS):
    """ plot of the correlation function"""

    ###############################################################################
    
    PROTOCOLS = ['unbalanced activity', 'proximal activity', 'distal activity', 'synchrony']
    
    YLABELS1 = [r"$\langle \nu_\mathrm{out}\rangle$"]+\
               [r"gain (Hz)"]+\
               ['gain increase (%) \n for '+p for p in PROTOCOLS]
    
    E_LABELS = [r"$\langle V_\mathrm{thre}^\mathrm{eff} \rangle_\mathcal{D}$ (mV)",\
                r"$\langle \partial \nu / \partial \mu_V \rangle_\mathcal{D}$ (Hz/mV)",\
                r"$\langle \partial \nu / \partial \sigma_V \rangle_\mathcal{D}$ (Hz/mV)",\
                r"$\langle \partial \nu / \partial \tau_V^{N} \rangle_\mathcal{D}$ (Hz/%)"]

    X = [BIOPHYSICS[i,:] for i in range(BIOPHYSICS.shape[0])]
    Y = [COUPLINGS[i,:] for i in range(COUPLINGS.shape[0])]

    INDEXES, MARKER, SIZE = range(4), ['^', 'd', '*', 's'], [12, 11, 17, 10]

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


if __name__=='__main__':

    F = np.linspace(.01, .6, 4)
    
    import matplotlib.pylab as plt
    sys.path.append('/home/yann/work/python_library/')
    from my_graph import set_plot, put_list_of_figs_to_svg_fig

    COUPLINGS = np.zeros((6, len(ALL_CELLS)))
    BIOPHYSICS = np.zeros((4, len(ALL_CELLS)))
    for i_nrn in range(len(ALL_CELLS)):
        print 'cell', i_nrn+1
        COUPLINGS[:,i_nrn] = single_experiment(F, i_nrn)
        BIOPHYSICS[:,i_nrn] = ALL_CELLS[i_nrn]['E']        

    fig = correlating_electrophy_and_coupling(COUPLINGS, BIOPHYSICS)
    fig.savefig('correlation.svg')
    plt.show()

