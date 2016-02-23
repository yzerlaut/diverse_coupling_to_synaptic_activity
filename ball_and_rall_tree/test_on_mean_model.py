from demo import *
import sys

soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')

if sys.argv[-1]=='run':
    tstop = 10000.

    x_exp, cables = setup_model(soma, stick, params)    

    x_stick = np.linspace(0,stick['L'],30)
    x_stick = .5*(x_stick[1:]+x_stick[:-1])

    # constructing the space-dependent shotnoise input for the simulation

    F = 0.2
    synch = 0. # baseline synchrony
    inh_factor = 7.
    shotnoise_input = {'synchrony':synch,
                       'fe_prox':F,'fi_prox':inh_factor*F,
                       'fe_dist':F,'fi_dist':inh_factor*F}

    print 'Running simulation [...]'
    t, V = run_simulation(shotnoise_input, cables, params, tstop=tstop)
    muV_exp, sV_exp, Tv_exp = analyze_simulation(x_exp, cables, t, V)
    np.save('data_mean_model_sim.npy', [x_exp, shotnoise_input, muV_exp, sV_exp, Tv_exp])
    fig = plot_time_traces(t, V, cables, params['EqCylinder'])
    fig.savefig('fig.svg', format='svg')
    plt.show()
    
else:
    x_exp, shotnoise_input, muV_exp, sV_exp, Tv_exp = \
      np.load('data_mean_model_sim.npy')
      
    # constructing the space-dependent shotnoise input for the simulation
    x_th, muV_th, sV_th, Tv_th  = \
                get_analytical_estimate(shotnoise_input,
                                        soma, stick, params,
                                        discret=20)
    make_comparison_plot(x_th, muV_th, sV_th, Tv_th,\
                         x_exp, muV_exp, sV_exp, Tv_exp, shotnoise_input)    
    plt.show()

