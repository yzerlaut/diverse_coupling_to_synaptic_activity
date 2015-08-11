import numpy as np
import matplotlib
import matplotlib.pylab as plt
import sys
sys.path.append('/home/yann/work/yns_python_library')
# from my_graph import set_plot
import fourier_for_real as rfft
from signanalysis import autocorrel

# ------- model parameters in SI units ------- # 
params = {
'g_pas': 1e-3*1e4, 'cm' : 1*1e-2, 'Ra' : 35.4*1e-2, 'El': -65e-3,\
'Qe' : 1.e-9 , 'Te' : 5.e-3, 'Ee': 0e-3,\
'Qi' : 1.5e-9 , 'Ti' : 5.e-3, 'Ei': -80e-3,\
'seed' : 0}

sim_params = {#in ms
'dt':0.025,
'initial_discard':200.,
'window_for_autocorrel':40.,
'tstop':5000.
}


FACTOR_FOR_DENSITY = 1. # because Ball & Sticks sucks !!!

# synaptic density  area (m2) per one synapse !!
# ball (soma)
soma = {'L': 40*1e-6, 'D': 20*1e-6, 'NSEG': 1,
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

def setup_model(soma, stick, params):
    params_for_cable_theory(stick, params)
    cables = [soma, stick]
    # density of synapses in um2, one synapses per this area !
    # so calculate synapse number
    Ke_tot, Ki_tot = 0, 0
    for i in range(len(cables)):
        cables[i]['Ki_per_seg'] = cables[i]['L']*\
              cables[i]['D']*np.pi/cables[i]['NSEG']/cables[i]['inh_density']
        Ki_tot += cables[i]['Ki_per_seg']*cables[i]['NSEG']
        cables[i]['Ke_per_seg'] = cables[i]['L']*\
              cables[i]['D']*np.pi/cables[i]['NSEG']/cables[i]['exc_density']
        Ke_tot += cables[i]['Ke_per_seg']*cables[i]['NSEG']
    print "Total number of EXCITATORY synapses : ", Ke_tot
    print "Total number of INHIBITORY synapses : ", Ki_tot
    return cables

from numerical_simulations.nrn_simulations import *
from scipy.optimize import curve_fit


def run_simulation(fe, fi, cables, params):

    exc_synapses, exc_netcons, exc_netstims,\
           inh_synapses, inh_netcons, inh_netstims,\
           area_lists, spkout =\
              Constructing_the_ball_and_tree(params, cables,
                                    spiking_mech=False)
    for i in range(len(cables)):
        for j in range(len(area_lists[i])):
            # excitation
            Ke = cables[i]['Ke_per_seg']
            exc_netstims[i][j].interval = 1e3/fe/Ke
            Ki = cables[i]['Ki_per_seg']
            inh_netstims[i][j].interval = 1e3/fi/Ki

    ## --- recording
    t_vec = nrn.Vector()
    t_vec.record(nrn._ref_t)
    ## --- launching the simulation
    nrn.finitialize(params['El']*1e3)
    nrn.dt = sim_params['dt']
    V = np.zeros((int(sim_params['tstop']/sim_params['dt']),cables[1]['NSEG']))
    i=0
    while nrn.t<(sim_params['tstop']-sim_params['dt']):
        j = 0
        for seg in nrn.cable_1_0:
            V[i,j] = nrn.cable_1_0(seg.x)._ref_v[0]
            j+=1
        i+=1
        nrn.fadvance()

    return np.array(t_vec), V

def analyze_simulation(t_vec, V):

    x = np.linspace(0, stick['L'], stick['NSEG'], endpoint=True)

    exp_f = lambda t, tau: np.exp(-t/tau)
    def find_acf_time(v_acf, t_shift, criteria=0.01):
        i_max = np.argmin(np.abs(v_acf-criteria))
        P, pcov = curve_fit(exp_f, t_shift[:i_max], v_acf[:i_max])
        return P[0]

    muV_exp, sV_exp = V[1000:,:].mean(axis=0), V[1000:,:].std(axis=0)
    Tv_exp = 0*muV_exp

    i0 = int(sim_params['initial_discard']/sim_params['dt']) # start for det of ACF
    for i in range(len(muV_exp)):
        v_acf, t_shift = autocorrel(V[i0:,i],\
            sim_params['window_for_autocorrel']*1e-3, 1e-3*sim_params['dt'])
        Tv_exp[i] = find_acf_time(v_acf, t_shift, criteria=0.01)

    return x, muV_exp, sV_exp, Tv_exp



def get_analytical_estimate(fe, fi, soma, stick, params, discret=20):

    print '----------------------------------------------------'
    print ' Analytical calculus running [...]'

    x_th = np.linspace(0, stick['L'], discret)

    Garray = calculate_mean_conductances(fe, fi, soma, stick, params)

    dt, tstop = 1e-5, 50e-3
    t = np.arange(int(tstop/dt))*dt
    f = rfft.time_to_freq(len(t), dt)
    muV_th = stat_pot_function(x_th, Garray, soma, stick, params)
    # sV_th = np.sqrt(get_the_theoretical_variance(fe, fi, f, x_th, params, soma, stick, precision=discret))
    # Tv_th = 0*sV_th
    sV_th, Tv_th = get_the_theoretical_sV_and_Tv(\
            fe, fi, f, x_th, params, soma, stick, precision=discret)
    # Rin, Rtf = get_the_input_and_transfer_resistance(fe, fi, f, x_th, params, soma, stick)

    print '----------------------------------------------------'
    print ' => end calculus'
    print '----------------------------------------------------'

    return x_th, muV_th, sV_th, Tv_th


def plot_time_traces(t_vec, V, title=''):
    V = np.array(V)
    t = np.array(t_vec)#[:-1]
    
    fig2 = plt.figure()
    plt.title(title)
    i1, i2 = 0, min([int(1000/(t[1]-t[0])),int(sim_params['tstop']/(t[1]-t[0]))])#int(tstop/dt/3.), int(2*tstop/dt/3.)
    plt.plot(1e-3*t[i1:i2], V[i1:i2,0], 'b', label='soma')
    plt.plot(1e-3*t[i1:i2], V[i1:i2,-1], 'r', label='distal')
    plt.plot(1e-3*t[i1:i2], V[i1:i2,int(V.shape[1]/2.)], 'g', label='medial')
    
    plt.xlabel('time (s)')
    plt.ylabel('$V_m$ (mV)')
    plt.legend(loc='best', prop={'size':'small'})
    plt.tight_layout()
    plt.savefig('./fig2.svg', format='svg')

    return fig2

def make_comparison_plot(x_th, muV_th, sV_th, Tv_th,\
                         x_exp, muV_exp, sV_exp, Tv_exp):

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
    AX[0].legend(loc='best', prop={'size':'small'})
    AX[2].set_xlabel('distance from soma ($\mu$m)')

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
    parser.add_argument("--fe", type=float,\
                        help="excitatory synaptic frequency",
                        default=5.)
    parser.add_argument("--fi", type=float,\
                        help="excitatory synaptic frequency",
                        default=20.)
    parser.add_argument("--discret", type=int,\
                        help="discretization for theoretical evaluation",
                        default=20.)
    # ball and stick properties
    parser.add_argument("--L_stick", type=float, help="Length of the stick in micrometer", default=2000.)
    parser.add_argument("--D_stick", type=float, help="Diameter of the stick", default=2.)
    # synaptic properties
    parser.add_argument("--Qe", type=float, help="Excitatory synaptic weight (nS)", default=1.)
    parser.add_argument("--Qi", type=float, help="Inhibitory synaptic weight (nS)", default=3.)

    args = parser.parse_args()
    # setting up the stick properties
    stick['L'] = args.L_stick*1e-6
    stick['D'] = args.D_stick*1e-6
    # settign up the synaptic properties
    params['Qe'] = args.Qe*1e-9
    params['Qi'] = args.Qi*1e-9

    print ' first we set up the model [...]'
    cables = setup_model(soma, stick, params)    

    # then we run the simulation if needed
    if args.simulation:
        print 'Running simulation [...]'
        t, V = run_simulation(args.fe, args.fi, cables, params)
        x_exp, muV_exp, sV_exp, Tv_exp = analyze_simulation(t, V)
        print 'saving the data as :', "data/fe_%1.2f_fi_%1.2f.npy" % (args.fe,args.fi)
        np.save("data/fe_%1.2f_fi_%1.2f.npy" % (args.fe,args.fi),\
                [x_exp, muV_exp, sV_exp, Tv_exp])
        plot_time_traces(t, V, title='$\\nu_e$=  %1.2f Hz, $\\nu_i$= %1.2f Hz' % (args.fe,args.fi))
        plt.show()

    # now analytical calculus
    x_th, muV_th, sV_th, Tv_th  = get_analytical_estimate(args.fe, args.fi,\
                         soma, stick, params, discret=args.discret)

    try:
        x_exp, muV_exp, sV_exp, Tv_exp = np.load("data/fe_%1.2f_fi_%1.2f.npy" % (args.fe,args.fi))
    except IOError:
        print '======================================================'
        print 'no numerical data available !!!  '
        print 'you should run the simulation with the --simulation option !' 
        print '======================================================'
        x_exp, muV_exp, sV_exp, Tv_exp = x_th.mean()+0*x_th,\
          muV_th.mean()+0*x_th, sV_th.mean()+0*x_th, Tv_th.mean()+0*x_th
    
    make_comparison_plot(x_th, muV_th, sV_th, Tv_th,\
                         x_exp, muV_exp, sV_exp, Tv_exp)    
    plt.show()
