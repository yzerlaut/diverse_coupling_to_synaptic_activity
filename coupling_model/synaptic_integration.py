import numpy as np
import sys
sys.path.append('../')
from theory.analytical_calculus import get_the_fluct_prop_at_soma

ALL_CELLS = np.load('../ball_and_rall_tree/all_cell_params.npy')

def get_fluct_var(i_nrn, F, exp_type='non specific activity'):

    if exp_type=='non specific activity':
        feG, fiG, feI, fiI = F, 5.*F, F, 5.*F # symmetric stimulation
    elif exp_type=='balance rupture':
        feG, fiG, feI, fiI = F, 3.*F, F, 3.*F # symmetric stimulation
    elif exp_type=='proximal activity':
        feG, fiG, feI, fiI = 2.*F, 10.*F, .5*F, 2.5*F  # symmetric stimulation
    elif exp_type=='distal activity':
        feG, fiG, feI, fiI = .5*F, 2.5*F, 2.*F, 10.*F # symmetric stimulation
    elif exp_type=='synchronized activity':
        feG, fiG, feI, fiI = F, 5.*F, F, 5.*F # symmetric stimulation
    else:
        print '------------------------------------------'
        print 'problem with the protocol: ', exp_type
        print '------------------------------------------'

    shtn_input = {'fi_soma':feG, 'fe_prox':feG,'fi_prox':fiG,
                  'fe_dist':feI,'fi_dist':fiI}

    muV, sV, TvN, muGn = get_the_fluct_prop_at_soma(shtn_input,\
       ALL_CELLS[i_nrn]['params'], ALL_CELLS[i_nrn]['soma'],\
            ALL_CELLS[i_nrn]['stick'])

    return feG, fiG, feI, fiI, muV, sV, TvN, muGn

def sine(t, w, t0=0):
    return np.sin(2.*np.pi*(t-t0)*w)

sigmoid = lambda x: 1./(1.+np.exp(-x))


if __name__=='__main__':

    import matplotlib.pylab as plt
    sys.path.append('/home/yann/work/python_library/')
    from my_graph import set_plot

    i_nrn = 2 # index of the neuron

    fig, AX = plt.subplots(4, 1, figsize=(4, 15))
    plt.subplots_adjust(left=.3, top=.8, wspace=.2, hspace=.2)
    F = np.linspace(0.1,30)
    COLORS=['r', 'b', 'g', 'c', 'k', 'm']

    PROTOCOLS = ['balance rupture', 'proximal activity', 'distal activity',\
                 'synchronized activity', 'non specific activity']
        
    for i in range(len(PROTOCOLS)):
        feG, fiG, feI, fiI, muV, sV, TvN, muGn = get_fluct_var(i_nrn, F,\
                                      exp_type=PROTOCOLS[i])
        for ax, x in zip(AX, [1e3*muV, 1e3*sV, 1e2*TvN, muGn]):
            ax.plot(F, x, lw=2, color=COLORS[i], label=PROTOCOLS[i])


    LABELS = ['$\mu_V$ (mV)', '$\sigma_V$ (mV)',\
              '$\\tau_V / \\tau_m^0$ (%)', '$\mu_G / g_L$']
    AX[0].legend(prop={'size':'small'}, bbox_to_anchor=(1., 2.5))
    for ax, ylabel in zip(AX[:-1], LABELS[:-1]):
        set_plot(ax, ['left'], ylabel=ylabel, xticks=[])
    set_plot(AX[-1], ['bottom','left'],\
             ylabel=LABELS[-1], xlabel='synaptic activity (Hz)')
    plt.show()
