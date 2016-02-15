import numpy as np
import sys
sys.path.append('../')
from theory.analytical_calculus import get_the_fluct_prop_at_soma
from synaptic_integration import get_fluct_var
from firing_response_description.template_and_fitting import final_func

## LOADING ALL CELLS PROPERTIES
soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')
ALL_CELLS = np.load('../input_impedance_calibration/all_cell_params.npy')

def single_experiment(F, i_nrn, exp_type='control', seed=1):
    
    feG, fiG, feI, fiI, synch, muV, sV, TvN, muGn = get_fluct_var(i_nrn, F, exp_type=exp_type)
    ## FIRING RATE RESPONSE
    Fout = final_func(ALL_CELLS[i_nrn]['P'], muV, sV, TvN,\
                      ALL_CELLS[i_nrn]['Gl'], ALL_CELLS[i_nrn]['Cm'])

    return Fout

if __name__=='__main__':

    print ALL_CELLS[0]
    # import matplotlib.pylab as plt
    # sys.path.append('/home/yann/work/python_library/')
    # from my_graph import set_plot, put_list_of_figs_to_svg_fig


    # F = np.linspace(.01, .6)

    # PROTOCOLS = ['unbalanced activity', 'proximal activity', 'distal activity',\
    #              'synchronized activity', 'non specific activity']

        
    # FIG_LIST = []
    # for i_nrn in range(4):
    #     fig, ax = plt.subplots(figsize=(4, 3))
    #     plt.subplots_adjust(left=.3, bottom=.3, wspace=.2, hspace=.2)
    #     print 'cell : ', i_nrn
    #     ax.set_title('cell'+str(i_nrn+1))
    #     for protocol, c in zip(PROTOCOLS, COLORS[:5]):
    #         Fout = single_experiment(F, i_nrn, exp_type=protocol)
    #         ax.plot(F, Fout, lw=4, color=c, label=protocol)
    #     if i_nrn==0:
    #         ax.legend(frameon=False, prop={'size':'xx-small'})
    #     set_plot(ax, xlabel='increasing \n presynaptic quantity', ylabel='$\\nu_{out}$ (Hz)', xticks=[])
    #     FIG_LIST.append(fig)
    #     fig.savefig('data/cell'+str(i_nrn+1)+'.svg')
    # put_list_of_figs_to_svg_fig(FIG_LIST)
