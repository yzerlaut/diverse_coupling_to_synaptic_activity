import numpy as np
import sys
sys.path.append('../')
from theory.analytical_calculus import get_the_fluct_prop_at_soma
from simple_synaptic_integration import get_fluct_var
from firing_response_description.template_and_fitting import final_func

## LOADING ALL CELLS PROPERTIES
soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')
ALL_CELLS = np.load('../input_impedance_calibration/all_cell_params.npy')

def single_experiment(i_nrn, exp_type='control', seed=1, len_f=5, balance=-54e-3, precision=1e2):
    
    feG, fiG, feI, fiI, synch, muV, sV, TvN, muGn = get_fluct_var(i_nrn, exp_type=exp_type, len_f=len_f,\
                                                                  balance=balance, precision=precision)
    ## FIRING RATE RESPONSE
    Fout = final_func(ALL_CELLS[i_nrn]['P'], muV, sV, TvN,\
                      ALL_CELLS[i_nrn]['Gl'], ALL_CELLS[i_nrn]['Cm'])

    return Fout

if __name__=='__main__':

    import matplotlib.pylab as plt
    sys.path.append('/home/yann/work/python_library/')
    from my_graph import set_plot, put_list_of_figs_to_svg_fig

    COLORS=['r', 'b', 'g', 'c', 'k', 'm']

    PROTOCOLS = ['unbalanced activity', 'proximal activity', 'distal activity',\
                 'synchrony', 'non specific activity']

    len_f = 20
    F = np.linspace(0,1, len_f)
                 
    FIG_LIST = []
    if sys.argv[-1]=='all':
        CHOOSED_CELLS = range(len(ALL_CELLS))
    else:
        CHOOSED_CELLS = [0, 2, 27, 1]
    ii=1
    for i_nrn in CHOOSED_CELLS:
        fig, ax = plt.subplots(figsize=(4, 3))
        plt.subplots_adjust(left=.3, bottom=.3, wspace=.2, hspace=.2)
        print 'cell : ', i_nrn+1
        ax.set_title('cell'+str(i_nrn+1))
        for protocol, c in zip(PROTOCOLS, COLORS[:5]):
            Fout = single_experiment(i_nrn, exp_type=protocol, len_f=len_f, precision=1e3)
            ax.plot(.5*(F[1:]+F[:-1]), .5*(Fout[1:]+Fout[:-1]), lw=4, color=c, label=protocol)
        if i_nrn==0:
            ax.legend(frameon=False, prop={'size':'xx-small'})
        set_plot(ax, xlabel='increasing \n presynaptic quantity', ylabel='$\\nu_{out}$ (Hz)', xticks=[])
        fig.savefig('data/cell'+str(i_nrn+1)+'.svg')
        if len(CHOOSED_CELLS)<len(ALL_CELLS):
            FIG_LIST.append(fig)
        else:
            plt.clf()
        ii+=1
    if sys.argv[-1]!='all':
        put_list_of_figs_to_svg_fig(FIG_LIST)
