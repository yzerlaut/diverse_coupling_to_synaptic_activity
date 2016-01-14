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

soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')
EqCylinder = np.linspace(0, 1, stick['B']+1)*stick['L']

# fixing the synaptic densities !!
FACTOR = 1./2. # we double the densities
soma['exc_density'], soma['inh_density']= 1e9, FACTOR*25.*1e-12
stick['exc_density'], stick['inh_density']= FACTOR*17*1e-12, FACTOR*100*1e-12


if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser(description=
     """ 
     description of the whole modulus here
     """
    ,formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-S", "--simulation",\
                        help="With numerical simulation (NEURON)",
                        action="store_true")
    parser.add_argument("--fe_prox", type=float, help="excitatory synaptic frequency in proximal compartment", default=1.)
    parser.add_argument("--fi_prox", type=float, help="inhibitory synaptic frequency in proximal compartment", default=5.)
    parser.add_argument("--fe_dist", type=float, help="excitatory synaptic frequency in distal compartment", default=1.)
    parser.add_argument("--fi_dist", type=float, help="inhibitory synaptic frequency in distal compartment", default=5.)
    parser.add_argument("--fe_soma", type=float, help="excitatory synaptic frequency at soma compartment", default=.0001)
    parser.add_argument("--fi_soma", type=float, help="inhibitory synaptic frequency at soma compartment", default=5.)
    parser.add_argument("--discret_sim", type=int, help="space discretization for numerical simulation", default=20)
    parser.add_argument("--tstop_sim", type=float, help="max simulation time (s)", default=2.)
    parser.add_argument("--discret_th", type=int, help="discretization for theoretical evaluation",default=20)
    # ball and stick properties
    parser.add_argument("--L_stick", type=float, help="Length of the stick in micrometer", default=2000.)
    parser.add_argument("--L_proximal", type=float, help="Length of the proximal compartment", default=2000.)
    parser.add_argument("--D_stick", type=float, help="Diameter of the stick", default=2.)
    parser.add_argument("-B", "--branches", type=int, help="Number of branches (equally spaced)", default=1)
    parser.add_argument("--EqCylinder", help="Detailed branching morphology (e.g [0.,0.1,0.25, 0.7, 1.])", nargs='+', type=float, default=[])
    # synaptic properties
    parser.add_argument("--Qe", type=float, help="Excitatory synaptic weight (nS)", default=1.)
    parser.add_argument("--Qi", type=float, help="Inhibitory synaptic weight (nS)", default=3.)

    args = parser.parse_args()

    # setting up the stick properties
    stick['L'] = args.L_stick*1e-6
    stick['L_prox'] = args.L_proximal*1e-6
    stick['D'] = args.D_stick*1e-6
    if not len(args.EqCylinder):
        params['B'] = args.branches
        EqCylinder = np.linspace(0, 1, params['B']+1)*stick['L'] # equally space branches !
    else:
        EqCylinder = np.array(args.EqCylinder)*stick['L'] # detailed branching
        
    # settign up the synaptic properties
    params['Qe'] = args.Qe*1e-9
    params['Qi'] = args.Qi*1e-9

    print ' first we set up the model [...]'
    stick['NSEG'] = args.discret_sim
    x_exp, cables = setup_model(EqCylinder, soma, stick, params)    

    # we adjust L_proximal so that it falls inbetweee two segments
    args.L_stick *= 1e-6 # SI units
    args.L_proximal *= 1e-6 # SI units
    L_proximal = int(args.L_proximal/args.L_stick*args.discret_sim)*args.L_stick/args.discret_sim
    x_stick = np.linspace(0,args.L_stick, args.discret_sim+1) # then :
    x_stick = .5*(x_stick[1:]+x_stick[:-1])
    # constructing the space-dependent shotnoise input for the simulation
    fe, fi = [], []
    fe.append([0]) # no excitation on somatic compartment
    fi.append([args.fi_soma]) # inhibition on somatic compartment
    for cable in cables[1:]:
        fe.append([args.fe_prox if x<L_proximal else args.fe_dist for x in cable['x']])
        fi.append([args.fi_prox if x<L_proximal else args.fi_dist for x in cable['x']])
        
    shtn_input = {'fi_soma':args.fi_soma,
                  'fe_prox':args.fe_prox,'fi_prox':args.fi_prox,
                  'fe_dist':args.fe_dist,'fi_dist':args.fi_dist}

    muV, sV, Tv, muGn = get_the_fluct_prop_at_soma(shtn_input, EqCylinder,\
                                                   params, soma, stick)
    
    print 1e3*muV, 1e3*sV, 1e3*Tv, muGn
