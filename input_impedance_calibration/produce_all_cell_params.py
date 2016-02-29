import numpy as np
import sys
sys.path.append('../')
from input_impedance_calibration.get_calib import adjust_model_prop # where the core calculus lies
from firing_response_description.template_and_fitting import final_func
from theory.analytical_calculus import *

#### ================================================== ##
#### MEAN MODEL ###############################
#### ================================================== ##

soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')
kept_cells = np.load('../coupling_model/kept_cells.npy')
from data_firing_response.analyze_data import get_Rm_range
Rm_exp = get_Rm_range()[kept_cells]

## SYNAPTIC SCALING

def Qe_rule(Rm, params):
    Qe0 = params['Qe']
    Rm0 = 250e6 # Mohm
    return Qe0*Rm0/Rm

def Qi_rule(Rm, params):
    Qi0 = params['Qi']
    Rm0 = 250e6 # Mohm
    return Qi0*Rm0/Rm


if sys.argv[-1]=='plot':

    sys.path.append('/home/yann/work/python_library/')
    from my_graph import *
    import matplotlib.pylab as plt
    
    fig1, ax = plt.subplots(1, figsize=(6,3))
    ax.fill_between([0,1],[0,0],np.ones(2)*(1e-5)**2/soma['inh_density'], color='r')
    ax.fill_between([0,1],[0,0],np.ones(2)*(1e-5)**2/soma['exc_density'], color='g', lw=4)
    ax.fill_between([3,4],[0,0],np.ones(2)*(1e-5)**2/stick['exc_density'], color='g')
    ax.fill_between([3,4],[0,0],np.ones(2)*(1e-5)**2/stick['inh_density'], color='r')
    set_plot(ax, ['left'], ylabel='synaptic densities \n synapses/100$\mu m^2$',\
             xticks=[], yticks=[0,15,30])
    # set_plot(ax[1], ['left'], ylabel='synaptic densities \n synapses/100$\mu m^2$', xticks=[])

    fig2, ax = plt.subplots(1,2, figsize=(8,3))
    fig2.suptitle('synaptic event')
    t = np.linspace(-5,25, 1e4)*1e-3
    g = lambda Q,tau: Q*np.array([np.exp(-tt/tau) if tt>0 else 0 for tt in t])
    Qe, Qi = Qe_rule(Rm_exp.mean()*1e6, params), Qi_rule(Rm_exp.mean()*1e6, params)
    ax[0].plot(1e3*t, 1e9*g(Qi, params['Ti']), 'r-', lw=3)
    ax[0].plot(1e3*t, 1e9*g(Qe, params['Te']), 'g-', lw=3)
    ax[0].set_title('somatic and proximal')
    set_plot(ax[0], ylabel='G (nS)', xticks=[0,15,30], xlabel='time (ms)', ylim=[0,4], yticks=[0,2,4])

    ax[1].plot(1e3*t, 1e9*g(params['factor_for_distal_synapses_weight']*Qi, params['factor_for_distal_synapses_tau']*params['Ti']), 'r-', lw=3, label='inh.')
    ax[1].plot(1e3*t, 1e9*g(params['factor_for_distal_synapses_weight']*Qe, params['factor_for_distal_synapses_tau']*params['Te']), 'g-', lw=3, label='exc.')
    ax[1].set_title('distal')
    ax[1].legend()
    set_plot(ax[1], ylabel='G (nS)', xticks=[0,15,30], xlabel='time (ms)', ylim=[0,4], yticks=[0,2,4])

    fig3, ax = plt.subplots(2, 1, figsize=(3,4))
    plt.subplots_adjust(left=.3, bottom=.3)
    Rm0 = np.linspace(0.9*Rm_exp.min(), 1.2*Rm_exp.max(), 20)
    ax[0].plot(Rm0, 1e9*Qe_rule(Rm0*1e6, params), 'g', lw=2)
    ax[1].plot(Rm0, 1e9*Qi_rule(Rm0*1e6, params), 'r', lw=2)
    set_plot(ax[0], ['left'], ylabel='$Q_e$ (nS)', xticks=[])
    set_plot(ax[1], ylabel='$Q_i$ (nS)', xlabel='somatic input \n resistance $(M\Omega)$',\
             xticks=[200,400,600], yticks=[1,2,3])
    plt.show()
else:
    # data of the reduced morphologies
    ALL_CELLS = np.load('../data_firing_response/reduced_data.npy')

    for cell in ALL_CELLS:
        cell['Rm'] = 1e-6/cell['Gl']
        soma1, stick1, params1 = adjust_model_prop(cell['Rm'], soma, stick, params2=params.copy())
        cell['R_tf_soma'] = get_the_mean_transfer_resistance_to_soma(soma1, stick1, params1)
        
        cell['soma'], cell['stick'], cell['params'] = soma1, stick1, params1
        cell['params']['Qe'] = Qe_rule(cell['R_tf_soma'], params)
        cell['params']['Qi'] = Qi_rule(cell['R_tf_soma'], params)

    np.save('all_cell_params.npy', ALL_CELLS)


