import numpy as np
import matplotlib
import matplotlib.pylab as plt
import sys
sys.path.append('/home/yann/work/yns_python_library')
# from my_graph import set_plot
import fourier_for_real as rfft
from signanalysis import autocorrel
import brt_drawing

# ------- model parameters in SI units ------- # 
params = {
    'g_pas': 1e-4*1e4, 'cm' : 1*1e-2, 'Ra' : 35.4*1e-2, 'El': -65e-3,\
    'Qe' : 1.e-9 , 'Te' : 5.e-3, 'Ee': 0e-3,\
    'Qi' : 1.5e-9 , 'Ti' : 5.e-3, 'Ei': -80e-3,\
    'seed' : 0}

sim_params = {#in ms
'dt':0.025,
'initial_discard':200.,
'window_for_autocorrel':40.,
'tstop':20000.
}


FACTOR_FOR_DENSITY = 1. # because Ball & Sticks sucks !!!

# synaptic density  area (m2) per one synapse !!
# ball (soma)
soma = {'L': 20*1e-6, 'D': 20*1e-6, 'NSEG': 1,
        'exc_density':FACTOR_FOR_DENSITY*1e9,
        'inh_density':FACTOR_FOR_DENSITY*25*1e-12,
        'Ke':1e-9, 'Ki':10., 'name':'soma'}
# stick
stick = {'L': 2000*1e-6, 'D': 4*1e-6, 'NSEG': 30,\
         'exc_density':FACTOR_FOR_DENSITY*17*1e-12,
         'inh_density':FACTOR_FOR_DENSITY*100*1e-12,
         'Ke':100, 'Ki':100., 'name':'dend'}

from analytical_calculus import *
# we calculate the parameters to plug into cable theory

def setup_model(EqCylinder, soma, dend, Params):
    """ returns the different diameters of the equivalent cylinder
    given a number of branches point"""
    cables, xtot = [], np.zeros(1)
    cables.append(soma.copy())
    cables[0]['inh_density'] = soma['inh_density']
    cables[0]['exc_density'] = soma['exc_density']
    Ke_tot, Ki_tot = 0, 0
    D = dend['D'] # mothers branch diameter
    for i in range(1,len(EqCylinder)):
        cable = dend.copy()
        cable['x1'], cable['x2'] = EqCylinder[i-1], EqCylinder[i]
        cable['L'] = cable['x2']-cable['x1']
        x = np.linspace(cable['x1'], cable['x2'], cable['NSEG']+1)
        cable['x'] = .5*(x[1:]+x[:-1])
        xtot = np.concatenate([xtot, cable['x']])
        cable['D'] = D*2**(-2*(i-1)/3.)
        cable['inh_density'] = dend['inh_density']
        cable['exc_density'] = dend['exc_density']
        cables.append(cable)

    Ke_tot, Ki_tot, jj = 0, 0, 0
    for cable in cables:
        cable['Ki_per_seg'] = cable['L']*\
          cable['D']*np.pi/cable['NSEG']/cable['inh_density']
        cable['Ke_per_seg'] = cable['L']*\
          cable['D']*np.pi/cable['NSEG']/cable['exc_density']
        # summing over duplicate of compartments
        Ki_tot += 2**jj*cable['Ki_per_seg']*cable['NSEG']
        Ke_tot += 2**jj*cable['Ke_per_seg']*cable['NSEG']
        if cable['name']!='soma':
            jj+=1
    print "Total number of EXCITATORY synapses : ", Ke_tot
    print "Total number of INHIBITORY synapses : ", Ki_tot
    return xtot, cables

from numerical_simulations.nrn_simulations import *
from scipy.optimize import curve_fit

def analyze_simulation(xtot, t_vec, V):

    muV_exp, sV_exp, Tv_exp = [], [], []

    # for autocorrelation analysis
    exp_f = lambda t, tau: np.exp(-t/tau)
    def find_acf_time(v_acf, t_shift, criteria=0.01):
        i_max = np.argmin(np.abs(v_acf-criteria))
        P, pcov = curve_fit(exp_f, t_shift[:i_max], v_acf[:i_max])
        return P[0]

    for i in range(len(cables)): # loop over levels
        n_level = max(1,2**(i-1)) # number of levels
        for k in range(cables[i]['NSEG']): # loop over segments first
            muV_exp.append(0)
            sV_exp.append(0)
            Tv_exp.append(0)
            # they all have the same discretization
            for j in range(n_level): # loop over branches then
                v = np.array([V[it][i][j][k] for it in range(int(.1*len(t)), len(t))]).flatten()
                muV_exp[-1] += v.mean()/n_level
                sV_exp[-1] += v.std()/n_level
                v_acf, t_shift = autocorrel(v,\
                  sim_params['window_for_autocorrel']*1e-3, 1e-3*sim_params['dt'])
                Tv_exp[-1] += find_acf_time(v_acf, t_shift, criteria=0.01)/n_level

    return np.array(muV_exp), np.array(sV_exp), np.array(Tv_exp)


def get_analytical_estimate(shotnoise_input, EqCylinder,
                            soma, stick, params, discret=20):

    print '----------------------------------------------------'
    print ' Analytical calculus running [...]'

    x_th = np.linspace(0, stick['L'], discret)
    
    dt, tstop = 1e-5, 50e-3
    t = np.arange(int(tstop/dt))*dt
    f = rfft.time_to_freq(len(t), dt)

    muV_th = 0*x_th
    muV_th = stat_pot_function(x_th, shotnoise_input, EqCylinder,
                               soma, stick, params)
    sV_th, Tv_th = 0*muV_th, 0*muV_th
    sV_th, Tv_th = get_the_theoretical_sV_and_Tv(shotnoise_input, EqCylinder, f,\
                                  x_th, params, soma, stick,
                                  precision=discret)
    # Rin, Rtf = get_the_input_and_transfer_resistance(fe, fi, f, x_th, params, soma, stick)

    print '----------------------------------------------------'
    print ' => end calculus'
    print '----------------------------------------------------'

    return x_th, muV_th, sV_th, Tv_th


def plot_time_traces(t_vec, V, cables, title=''):

    # time window
    i1, i2 = 0, min([int(1000/(t[1]-t[0])),len(t_vec)])

    # we define the points that we want to extract

    # ---- soma
    i_level0, i_branch0, frac_seg0 = 1, 0, 0.
    i_seg0 = int(frac_seg0*cables[i_level0]['NSEG'])
    v0 = np.array([V[i][i_level0][i_branch0][i_seg0] for i in range(i1,i2)])

    # ---- medial
    if len(cables)==2:
        i_level1, i_branch1 =0, 0
    else:
        i_level1, i_branch1= len(cables)/2+1, max([1,2**(len(cables)/2)-1])
    frac_seg1 =.5
    i_seg1 = int(frac_seg1*cables[i_level1]['NSEG'])
    v1 = np.array([V[i][i_level1][i_branch1][i_seg1] for i in range(i1,i2)])

    # ---- distal
    i_level2, i_branch2, frac_seg2 = len(cables)-1, 0, 1.
    i_seg2 = int(frac_seg2*cables[i_level2]['NSEG'])-1
    v2 = np.array([V[i][i_level2][i_branch2][i_seg2] for i in range(i1,i2)])
    
    ## first we draw the fig where we will see the inputs
    fig, ax = brt_drawing.make_fig(EqCylinder, cables[1]['D'],\
        added_points=[\
                      [i_level2, i_branch2+1, frac_seg2,'r','D', 10],\
                      [i_level1, i_branch1+1, frac_seg1,'g','D', 10],\
                      [i_level0, i_branch0+1, frac_seg0,'b','D', 10]])
    fig2 = plt.figure()

    plt.title(title)
    plt.plot(1e-3*t[i1:i2], v0, 'b', label='soma')
    plt.plot(1e-3*t[i1:i2], v1, 'g', label='medial')
    plt.plot(1e-3*t[i1:i2], v2, 'r', label='distal')
    
    plt.xlabel('time (s)')
    plt.ylabel('$V_m$ (mV)')
    plt.legend(loc='best', prop={'size':'small'})
    plt.tight_layout()
    plt.savefig('./fig2.svg', format='svg')

    return fig2

def make_comparison_plot(x_th, muV_th, sV_th, Tv_th,\
                      x_exp, muV_exp, sV_exp, Tv_exp, shotnoise_input):

    # input frequencies
    # fig, ax = plt.subplots(2, 1, figsize=(3,6))

    # membrane pot 
    fig1, AX = plt.subplots(3,1, sharex=True, figsize=(5,8))
    # numerical simulations
    AX[0].plot(1e6*x_exp, muV_exp, 'kD', label='num. sim')
    AX[1].plot(1e6*x_exp, sV_exp, 'kD')
    AX[2].plot(1e6*x_exp, 1e3*Tv_exp, 'kD')
    # analytical calculuc
    AX[0].plot(1e6*x_th, 1e3*muV_th, 'k-', label='analytic approx.', lw=3, alpha=.5)
    AX[1].plot(1e6*x_th, 1e3*sV_th, 'k-', lw=3, alpha=.5)
    AX[2].plot(1e6*x_th, 1e3*Tv_th, 'k-', lw=3, alpha=.5)
    AX[0].legend(loc='best', prop={'size':'small'}, frameon=False)
    AX[2].set_xlabel('distance from soma ($\mu$m)')
    AX[2].set_ylim([min(1e3*Tv_exp.min()-1,1e3*Tv_th.min()-1),
                   max(1e3*Tv_exp.max()+1,1e3*Tv_th.max()+1)])

    YLABELS = ['$\mu_V$ (mV)', '$\sigma_V$ (mV)', '$\tau_V$ (ms)']
    # for ax, ylabel in zip([AX, YLABELS]):
    #     set_plot(ax, ylabel=ylabel)

    plt.tight_layout()
    plt.savefig('./fig1.svg', format='svg')
    
    return fig1

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
    parser.add_argument("--fe_prox", type=float, help="excitatory synaptic frequency in proximal compartment", default=5.)
    parser.add_argument("--fi_prox", type=float, help="inhibitory synaptic frequency in proximal compartment", default=20.)
    parser.add_argument("--fe_dist", type=float, help="excitatory synaptic frequency in distal compartment", default=5.)
    parser.add_argument("--fi_dist", type=float, help="inhibitory synaptic frequency in distal compartment", default=20.)
    parser.add_argument("--fe_soma", type=float, help="excitatory synaptic frequency at soma compartment", default=.0001)
    parser.add_argument("--fi_soma", type=float, help="inhibitory synaptic frequency at soma compartment", default=20.)
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
    # then we run the simulation if needed
    if args.simulation:
        print 'Running simulation [...]'
        t, V = run_simulation(fe, fi, cables, params, tstop=args.tstop_sim*1e3, dt=0.025)
        muV_exp, sV_exp, Tv_exp = analyze_simulation(x_exp, t, V)
        print 'saving the data as :', "data/fe_prox_%1.2f_fe_dist_%1.2f_fi_prox_%1.2f_fi_dist_%1.2f.npy" % (args.fe_prox,args.fe_dist,args.fi_prox,args.fi_prox)
        np.save("data/fe_prox_%1.2f_fe_dist_%1.2f_fi_prox_%1.2f_fi_dist_%1.2f.npy" % (args.fe_prox,args.fe_dist,args.fi_prox,args.fi_prox),\
                [x_exp, fe, fi, muV_exp, sV_exp, Tv_exp])
        # now plotting of simulated membrane potential traces
        plot_time_traces(t, V, cables,\
            title='$\\nu_e^p$=  %1.2f Hz, $\\nu_e^d$=  %1.2f Hz, $\\nu^p_i$= %1.2f Hz, $\\nu^d_i$= %1.2f Hz' % (args.fe_prox,args.fe_dist,args.fi_prox,args.fi_prox))
        plt.show()

    shotnoise_input = {'fi_soma':args.fi_soma,
                       'fe_prox':args.fe_prox,'fi_prox':args.fi_prox,
                       'fe_dist':args.fe_dist,'fi_dist':args.fi_dist}

    # ===== now analytical calculus ========

    # constructing the space-dependent shotnoise input for the simulation
    stick['L_prox'] = L_proximal

    # constructing the space-dependent shotnoise input for the simulation
    x_th, muV_th, sV_th, Tv_th  = \
                get_analytical_estimate(shotnoise_input, EqCylinder,
                                        soma, stick, params,
                                        discret=args.discret_th)
    try:
        x_exp, fe_exp, fi_exp, muV_exp, sV_exp, Tv_exp = np.load("data/fe_prox_%1.2f_fe_dist_%1.2f_fi_prox_%1.2f_fi_dist_%1.2f.npy" %  (args.fe_prox,args.fe_dist,args.fi_prox,args.fi_prox))
    except IOError:
        print '======================================================'
        print 'no numerical data available !!!  '
        print 'you should run the simulation with the --simulation option !' 
        print '======================================================'
        x_exp, muV_exp, sV_exp, Tv_exp = x_th.mean()+0*x_th,\
          muV_th.mean()+0*x_th, sV_th.mean()+0*x_th, Tv_th.mean()+0*x_th
    
    make_comparison_plot(x_th, muV_th, sV_th, Tv_th,\
                         x_exp, muV_exp, sV_exp, Tv_exp, shotnoise_input)    
    plt.show()
