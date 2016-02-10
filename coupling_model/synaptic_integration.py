import numpy as np
import sys
sys.path.append('../')
from theory.analytical_calculus import get_the_fluct_prop_at_soma, find_balance_at_soma

soma, stick, params = np.load('../ball_and_rall_tree/mean_model.npy')
ALL_CELLS = np.load('../ball_and_rall_tree/all_cell_params.npy')

def find_inh_cond_for_balance(feG, fiG, feI, fiI, i_nrn, balance=-60e-3):
    for i in range(len(F)):
        fiG[i], fiI[i] = find_balance_at_soma(feG[i], feI[i],\
                    ALL_CELLS[i_nrn]['params'], ALL_CELLS[i_nrn]['soma'],\
                    ALL_CELLS[i_nrn]['stick'], balance=balance)
    return fiG, fiI

def get_fluct_var(i_nrn, F, exp_type='non specific activity', balance=-58e-3):

    synch = 0.5 # baseline synchrony
    EI = 0.25 # for the excitatory/inhibitory balance
    EI_rupt = 0.4 # unbalanced activity
    PFrac = 1.
    DFrac = 1.
    if exp_type=='non specific activity':
        feG, fiG, feI, fiI = F, 0.*F, F, 0.*F
        fiG, fiI = find_inh_cond_for_balance(feG, fiG, feI, fiI, i_nrn, balance=balance)
    elif exp_type=='unbalanced activity':
        feG, fiG, feI, fiI = EI*F, .7*(1-EI)*F, EI*F, .7*(1-EI)*F
    elif exp_type=='proximal activity':
        feG, fiG, feI, fiI = 2.*F, 0.*F, 0*F, 0.*F
        fiG, fiI = find_inh_cond_for_balance(feG, fiG, feI, fiI, i_nrn, balance=balance)
    elif exp_type=='distal activity':
        feG, fiG, feI, fiI = 0*F, 0.*F, 2.*F, 0.*F
        fiG, fiI = find_inh_cond_for_balance(feG, fiG, feI, fiI, i_nrn, balance=balance)
    elif exp_type=='synchronized activity':
        feG, fiG, feI, fiI = EI*F, (1-EI)*F, EI*F, (1-EI)*F
        synch = 0.8
    else:
        print '------------------------------------------'
        print 'problem with the protocol: ', exp_type
        print '------------------------------------------'

    shtn_input = {'fi_soma':fiG, 'fe_prox':feG,'fi_prox':fiG,
                  'fe_dist':feI,'fi_dist':fiI, 'synchrony':synch+0.*F}

    muV, sV, TvN, muGn = get_the_fluct_prop_at_soma(shtn_input,\
       ALL_CELLS[i_nrn]['params'], ALL_CELLS[i_nrn]['soma'],\
            ALL_CELLS[i_nrn]['stick'])

    return feG, fiG, feI, fiI, synch, muV, sV, TvN, muGn

def sine(t, w, t0=0):
    return np.sin(2.*np.pi*(t-t0)*w)

sigmoid = lambda x: 1./(1.+np.exp(-x))


if __name__=='__main__':

    import matplotlib.pylab as plt
    sys.path.append('/home/yann/work/python_library/')
    from my_graph import set_plot

    i_nrn = 2 # index of the neuron

    fig, AX = plt.subplots(4, 1, figsize=(4, 15))
    fig2, AX2 = plt.subplots(4, 1, figsize=(4, 15))
    plt.subplots_adjust(left=.3, top=.8, wspace=.2, hspace=.2)
    F = np.linspace(.001,1.5,3)
    COLORS=['r', 'b', 'g', 'c', 'k', 'm']

    PROTOCOLS = ['unbalanced activity', 'proximal activity', 'distal activity',\
                 'synchronized activity', 'non specific activity']
        
    for i in range(len(PROTOCOLS)):
        if sys.argv[-1]=='all':
            for i_nrn in range(len(ALL_CELLS)):
                print 'cell: ', i_nrn
                feG, fiG, feI, fiI, synch, muV, sV, TvN, muGn = get_fluct_var(i_nrn, F,\
                                              exp_type=PROTOCOLS[i])
                for ax, x in zip(AX, [1e3*muV, 1e3*sV, 1e2*TvN, muGn]):
                    ax.plot(F, x, lw=.5, color=COLORS[i], label=PROTOCOLS[i])
                for ax, x in zip(AX2, [feG, fiG, feI, fiI]):
                    ax.plot(F, x, lw=.5, color=COLORS[i], label=PROTOCOLS[i])
        else:
            feG, fiG, feI, fiI, synch, muV, sV, TvN, muGn = get_fluct_var(i_nrn, F,\
                                          exp_type=PROTOCOLS[i])
            for ax, x in zip(AX, [1e3*muV, 1e3*sV, 1e2*TvN, muGn]):
                ax.plot(F, x, lw=2, color=COLORS[i], label=PROTOCOLS[i])
            for ax, x in zip(AX2, [feG, fiG, feI, fiI]):
                ax.plot(F, x, lw=.5, color=COLORS[i], label=PROTOCOLS[i])

    LABELS = ['$\mu_V$ (mV)', '$\sigma_V$ (mV)',\
              '$\\tau_V / \\tau_m^0$ (%)', '$g_{tot}^{soma} / g_L$']
    
    AX[0].legend(prop={'size':'small'}, bbox_to_anchor=(1., 1.))
    for ax, ylabel in zip(AX[:-1], LABELS[:-1]):
        set_plot(ax, ['left'], ylabel=ylabel, xticks=[])
    set_plot(AX[-1], ['bottom','left'],\
             ylabel=LABELS[-1], xlabel='synaptic activity (Hz)')
    plt.show()
