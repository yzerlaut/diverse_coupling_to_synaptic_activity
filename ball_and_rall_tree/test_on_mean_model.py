from demo import *

tstop = 2000.

soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')
EqCylinder = np.linspace(0, 1, stick['B']+1)*stick['L'] # equally space branches !

x_exp, cables = setup_model(EqCylinder, soma, stick, params)    

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
# now plotting of simulated membrane potential traces
plot_time_traces(t, V, cables, EqCylinder)
plt.show()

# constructing the space-dependent shotnoise input for the simulation
x_th, muV_th, sV_th, Tv_th  = \
            get_analytical_estimate(shotnoise_input, EqCylinder,
                                    soma, stick, params,
                                    discret=100)
make_comparison_plot(x_th, muV_th, sV_th, Tv_th,\
                     x_exp, muV_exp, sV_exp, Tv_exp, shotnoise_input)    
plt.show()

