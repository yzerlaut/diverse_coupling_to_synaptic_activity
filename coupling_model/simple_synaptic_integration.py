import numpy as np
import sys
sys.path.append('../')
from theory.analytical_calculus import *

soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')
ALL_CELLS = np.load('../input_impedance_calibration/all_cell_params.npy')

def find_inh_cond_for_balance(feG, fiG, feI, fiI, fe0, i_nrn, balance):
    for i in range(len(feG)):
        fiG[i], fiI[i] = find_balance_at_soma(feG[i], feI[i], fe0,\
                    ALL_CELLS[i_nrn]['params'], ALL_CELLS[i_nrn]['soma'],\
                    ALL_CELLS[i_nrn]['stick'], balance=balance[i])
    return fiG, fiI

def get_fluct_var(i_nrn, F=None, exp_type='non specific activity',\
                  balance=-55e-3, len_f=5):

    if F is None:
        F = np.linspace(0., .15, len_f)
        
    synch = 0.05+0*F # baseline synchrony
    F0 = 0.05+0*F
    
    inh_factor = 5.8
    inh_factor_balance_rupt = 4.
    
    fe0 = find_baseline_excitation(params, soma, stick,\
                                   balance=balance, synch=synch)
    
    if exp_type=='non specific activity':
        feG, fiG, feI, fiI = F+F0, inh_factor*(0*F+F0), F+F0, inh_factor*(0*F+F0)
        fiG, fiI = find_inh_cond_for_balance(feG, fiG, feI, fiI, fe0,i_nrn, balance+0*F)
    elif exp_type=='unbalanced activity':
        feG, feI, fiG, fiI = (F+F0), (F+F0), 0*F, 0*F
        fiG, fiI = find_inh_cond_for_balance(feG, fiG, feI, fiI, fe0,i_nrn,\
                                             balance+3e-3*np.linspace(0,1,len(F)))
    elif exp_type=='proximal activity':
        feG, fiG, feI, fiI = F0+4.*F, 4.*inh_factor*(F+F0), F0, 0*F
        fiG, fiI = find_inh_cond_for_balance(feG, fiG, feI, fiI, fe0,i_nrn, balance+0*F)
    elif exp_type=='distal activity':
        feG, fiG, feI, fiI = F0, 0*F, F0+2.*F, 2.*inh_factor*(F+F0)
        fiG, fiI = find_inh_cond_for_balance(feG, fiG, feI, fiI, fe0,i_nrn, balance+0*F)
    elif exp_type=='synchrony':
        synch = synch+np.linspace(0., 0.3, len(F))
        feG, feI, fiG, fiI = F0, F0, 0*F, 0*F
        fiG, fiI = find_inh_cond_for_balance(feG, fiG, feI, fiI, fe0,i_nrn, balance+0*F)
    else:
        print '------------------------------------------'
        print 'problem with the protocol: ', exp_type
        print '------------------------------------------'

    shtn_input = {'fi_soma':fiG, 'fe_prox':feG+fe0,'fi_prox':fiG,
                  'fe_dist':feI+fe0,'fi_dist':fiI, 'synchrony':synch+0.*F}

    muV, sV, TvN, muGn = get_the_fluct_prop_at_soma(shtn_input,\
       ALL_CELLS[i_nrn]['params'], ALL_CELLS[i_nrn]['soma'],\
            ALL_CELLS[i_nrn]['stick'])
            
    return feG+fe0, fiG, feI+fe0, fiI, synch, muV, sV, TvN, muGn

def sine(t, w, t0=0):
    return np.sin(2.*np.pi*(t-t0)*w)

sigmoid = lambda x: 1./(1.+np.exp(-x))

LABELS = ['$\mu_V$ (mV)', '$\sigma_V$ (mV)',\
          '$\\tau_V / \\tau_m^0$ (%)', '$g_{tot}^{soma} / g_L$']
LABELS2 = ['$\\nu_e^{prox}$ (Hz)', '$\\nu_i^{prox}$ (Hz)',\
           '$\\nu_e^{dist}$ (Hz)', '$\\nu_i^{dist}$ (Hz)',
           'synchrony']

if __name__=='__main__':

    import matplotlib.pylab as plt
    sys.path.append('/home/yann/work/python_library/')
    from my_graph import set_plot

    i_nrn = 2 # index of the neuron

    fig, AX = plt.subplots(4, 1, figsize=(4, 15))
    plt.subplots_adjust(left=.3, top=.8, wspace=.2, hspace=.2)
    fig2, AX2 = plt.subplots(5, 1, figsize=(4, 15))
    plt.subplots_adjust(left=.3, top=.8, wspace=.2, hspace=.2)
    COLORS=['r', 'b', 'g', 'c', 'k', 'm']

    PROTOCOLS = ['unbalanced activity', 'proximal activity', 'distal activity',\
                 'synchrony', 'non specific activity']
    len_f = 10
    F = np.linspace(0,1, len_f)
    if sys.argv[-1]=='all':

        FEG, FIG, FEI, FII, SYNCH, MUV, SV, TVN, MUGN, FOUT =\
           [np.zeros((len(PROTOCOLS), len(ALL_CELLS), len(F))) for j in range(10)]
           
        for i in range(len(PROTOCOLS)):

            for i_nrn in range(len(ALL_CELLS)):
                print 'cell: ', i_nrn
                FEG[i, i_nrn,:], FIG[i, i_nrn,:], FEI[i, i_nrn,:], FII[i, i_nrn,:], SYNCH[i, i_nrn,:],\
                   MUV[i, i_nrn,:], SV[i, i_nrn,:], TVN[i, i_nrn,:], MUGN[i, i_nrn,:] =\
                   get_fluct_var(i_nrn, exp_type=PROTOCOLS[i], len_f=len_f)

            np.save('data/synaptic_data.npy', [FEG, FIG, FEI, FII, SYNCH, MUV, SV, TVN, MUGN, FOUT])
            
    elif sys.argv[-1]=='plot':
                
        FEG, FIG, FEI, FII, SYNCH, MUV, SV, TVN, MUGN, FOUT = np.load('data/synaptic_data.npy')
            
        for i in range(len(PROTOCOLS)):
            for ax, x in zip(AX, [1e3*MUV[i,:,:], 1e3*SV[i,:,:], 1e2*TVN[i,:,:], MUGN[i,:,:]]):
                ax.plot(F, x.mean(axis=0), lw=3, color=COLORS[i])
                if sys.argv[-2]=='with_variations':
                    ax.fill_between(F, x.mean(axis=0)-x.std(axis=0),\
                                    x.mean(axis=0)+x.std(axis=0), alpha=.2, color=COLORS[i])
            for ax, x in zip(AX2, [FEG[i,:,:], FIG[i,:,:], FEI[i,:,:], FII[i,:,:], SYNCH[i,:,:]]):
                ax.plot(F, x.mean(axis=0), lw=3, color=COLORS[i])
                if sys.argv[-2]=='with_variations':
                    ax.fill_between(F, x.mean(axis=0)-x.std(axis=0),\
                                x.mean(axis=0)+x.std(axis=0), alpha=.2, color=COLORS[i])
    else:
        for i in range(len(PROTOCOLS)):
            feG, fiG, feI, fiI, synch, muV, sV, TvN, muGn = get_fluct_var(i_nrn,\
                                          exp_type=PROTOCOLS[i], len_f=len_f)
            for ax, x in zip(AX, [1e3*muV, 1e3*sV, 1e2*TvN, muGn]):
                ax.plot(F, x, lw=3, color=COLORS[i], label=PROTOCOLS[i])
            for ax, x in zip(AX2, [feG, fiG, feI, fiI, synch]):
                ax.plot(F, x, lw=3, color=COLORS[i], label=PROTOCOLS[i])

                
    for ax, ylabel in zip(AX[:-1], LABELS[:-1]):
        set_plot(ax, ['left'], ylabel=ylabel, xticks=[])
    for ax, ylabel in zip(AX2[:-1], LABELS2[:-1]):
        set_plot(ax, ['left'], ylabel=ylabel, xticks=[])
    set_plot(AX[-1], ['bottom','left'],\
             ylabel=LABELS[-1], xticks=[], xlabel='increasing \n presynaptic quantity')
    set_plot(AX2[-1], ['bottom','left'],\
             ylabel=LABELS2[-1], xticks=[], xlabel='increasing \n presynaptic quantity')

    if sys.argv[-1]!='all':
        AX[0].legend(prop={'size':'xx-small'}, bbox_to_anchor=(1., 2.))
        
    fig.savefig('fig.svg')
    fig2.savefig('fig2.svg')
             
