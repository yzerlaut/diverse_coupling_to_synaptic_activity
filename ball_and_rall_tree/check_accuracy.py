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

def build_paradigms(args):
    return [\
             {'label':'increasing excitation',
              'xlabel':'$\\nu_e^p=\\nu_e^d$ (Hz)',
              'vec':np.array([.7,1.,1.3]),
              'ref':args.fe_dist,
              'actions':"""
              s2['fe_prox']=[vec[0]*args.fe_prox,vec[1]*args.fe_prox]
              s2['fe_dist']=[vec[0]*args.fe_dist, vec[1]*args.fe_dist]"""
             },
             {'label':'increasing inhibition',
              'xlabel':'$\\nu_i^p=\\nu_i^d$ (Hz)',
              'vec':np.array([.7,1.,1.3]),
              'ref':args.fi_dist,
              'actions':"""
              s2['fi_prox']=[vec[0]*args.fi_prox,vec[1]*args.fi_prox]
              s2['fi_dist']=[vec[0]*args.fi_dist[vec[1]*args.fi_dist]"""
             },
             {'label':'increasing proximal act.',
              'xlabel':r'$\nu_i^p \propto \nu_e^p$ (Hz)',
              'vec':np.array([.7,1.,1.3]),
              'ref':(.8*args.fe_prox+.2*args.fi_prox),
              'actions':"""
              s2['fi_prox']=[vec[0]*args.fi_prox,vec[1]*args.fi_prox]
              s2['fe_prox']=[vec[0]*args.fe_proxvec[1]*args.fe_prox]"""
             },
            {'label':'increasing distal act.',\
              'xlabel':r'$\nu_i^d \propto \nu_e^d$ (Hz)',
              'vec':np.array([.7,1.,1.3]),
              'ref':(.8*args.fe_dist+.2*args.fi_dist),
              'actions':
                  """
                  s2['fi_dist']=[vec[0]*args.fi_dist,vec[1]*args.fi_dist]
                  s2['fe_dist']=[vec[0]*args.fe_dist,vec[1]*args.fe_dist]"""
             },
             {'label':'increasing synchrony',
              'xlabel':r'$\nu_i^d \propto \nu_e^d$ (Hz)',
              'vec':np.array([.7,1.,1.3]),
              'ref':(.8*args.fe_dist+.2*args.fi_dist),
              'actions':"s2['synchrony']=[vec[0]*args.synchrony,vec[1]*args.synchrony]"
             }]
             

if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser(description=
     """ 
     description of the whole modulus here
     """
    ,formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--fe_prox", type=float, help="excitatory synaptic frequency in proximal compartment", default=.5)
    parser.add_argument("--fi_prox", type=float, help="inhibitory synaptic frequency in proximal compartment", default=2.5)
    parser.add_argument("--fe_dist", type=float, help="excitatory synaptic frequency in distal compartment", default=.5)
    parser.add_argument("--fi_dist", type=float, help="inhibitory synaptic frequency in distal compartment", default=2.5)
    parser.add_argument("--synchrony", type=float, help="synchrony", default=.5)

    parser.add_argument("--VARIATIONS", action='store_true')
    parser.add_argument("--PLOT", action='store_true')
    parser.add_argument("--SIM", action='store_true')

    args = parser.parse_args()

    shtn_input = {'fi_soma':[args.fi_prox], 'synchrony':[args.synchrony],
                  'fe_prox':[args.fe_prox],'fi_prox':[args.fi_prox],
                  'fe_dist':[args.fe_dist],'fi_dist':[args.fi_dist]}

    # load mean cellular parameters
    soma, stick, params = np.load('mean_model.npy')
    muV0, sV0, TvN0, muGn0 = np.array(get_the_fluct_prop_at_soma(shtn_input,\
                                                    params, soma, stick))[:,0]
    if args.VARIATIONS:
        paradigms = build_paradigms(args)
        VALUES = []
        for p in paradigms:
            vec = p['vec']
            s2 = shtn_input.copy()
            exec(p['actions'])
            muV1, sV1, TvN1, muGn1 = 0.*vec, 0.*vec, 0.*vec, 0.*vec
            muV1, sV1, TvN1, muGn1 = np.array(get_the_fluct_prop_at_soma(s2, params, soma, stick))
            
        
        
        
    else:
        print 1e3*muV0, 1e3*sV0, 1e2*TvN0, muGn0
    # print '--- cell'
    # print 1e3*muV, 1e3*sV, TvN, muGn
    # print final_func(cell['P'], muV, sV, TvN, cell['Gl'], cell['Cm'])

