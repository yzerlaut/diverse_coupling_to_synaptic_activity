from demo import *
import sys

soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')
stick['NSEG'] = 20

inh_factor = 5.8
fe_baseline, fi_baseline, synch_baseline = 0.1, 0.1*inh_factor, 0.05
shtn_input = {'synchrony':synch_baseline,
                'fe_prox':fe_baseline, 'fi_prox':fi_baseline,
                  'fe_dist':2.*fe_baseline, 'fi_dist':fi_baseline}

if sys.argv[-1]=='run':
    tstop = 2000.

    x_exp, cables = setup_model(soma, stick, params)    
    x_stick = np.linspace(0,stick['L'],30)
    x_stick = .5*(x_stick[1:]+x_stick[:-1])

    # constructing the space-dependent shotnoise input for the simulation

    ## MAKING THE BASELINE EXPERIMENT
    print 'Running simulation [...]'
    t, V = run_simulation(shtn_input, cables, params, tstop=tstop, dt=0.05)
    np.save('data/data_mean_model_sim.npy', [x_exp, shtn_input, cables, t, V])
    
elif sys.argv[-1]=='analyze':
    x_exp, shtn_input, cables, t, V = \
      np.load('data/data_mean_model_sim.npy')
    x_exp, cables = setup_model(soma, stick, params)    
    print 'Analyzing data [...]'
    muV_exp, sV_exp, Tv_exp = analyze_simulation(x_exp, cables, t, V)

    np.save('data/analyzed_data_mean_model_sim.npy', [x_exp, muV_exp, sV_exp, Tv_exp, shtn_input, soma, stick, params])
    fig = plot_time_traces(t, V, cables, params['EqCylinder'])
    fig.savefig('fig.svg', format='svg')
    plt.show()
else:
    x_exp, muV_exp, sV_exp, Tv_exp, shtn_input, soma, stick, params = np.load('data/analyzed_data_mean_model_sim.npy')
                           
    # constructing the space-dependent shotnoise input for the simulation
    x_th, muV_th, sV_th, Tv_th  = \
                get_analytical_estimate(shtn_input, soma, stick, params,
                                        discret=20)
    make_comparison_plot(x_th, muV_th, sV_th, Tv_th,\
                         x_exp, muV_exp, sV_exp, Tv_exp, shtn_input)    
    plt.show()

