import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append('/home/yann/work/python_library')
from my_graph import set_plot
import fourier_for_real as rfft
sys.path.append('../')
from theory.analytical_calculus import *
from data_firing_response.analyze_data import get_Rm_range
from input_impedance_calibration.get_calib import adjust_model_prop

from firing_response_description.template_and_fitting import final_func

if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser(description=
     """ 
     description of the whole modulus here
     """
    ,formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--fe_prox", type=float, help="excitatory synaptic frequency in proximal compartment", default=1.)
    parser.add_argument("--fi_prox", type=float, help="inhibitory synaptic frequency in proximal compartment", default=5.)
    parser.add_argument("--fe_dist", type=float, help="excitatory synaptic frequency in distal compartment", default=1.)
    parser.add_argument("--fi_dist", type=float, help="inhibitory synaptic frequency in distal compartment", default=5.)
    parser.add_argument("--fe_soma", type=float, help="excitatory synaptic frequency at soma compartment", default=.0001)
    parser.add_argument("--fi_soma", type=float, help="inhibitory synaptic frequency at soma compartment", default=5.)

    args = parser.parse_args()

    shtn_input = {'fi_soma':[args.fi_soma],
                  'fe_prox':[args.fe_prox],'fi_prox':[args.fi_prox],
                  'fe_dist':[args.fe_dist],'fi_dist':[args.fi_dist]}

    ALL_CELLS = np.load('all_cell_params.npy')
    
    for cell in ALL_CELLS:
        muV, sV, TvN, muGn = get_the_fluct_prop_at_soma(shtn_input,\
                           cell['params'], cell['soma'], cell['stick'])
        print '--- cell'
        print 1e3*muV, 1e3*sV, TvN, muGn
        print final_func(cell['P'], muV, sV, TvN, cell['Gl'], cell['Cm'])

