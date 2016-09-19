import numpy as np
import matplotlib.pylab as plt
import os, sys
sys.path.append('/Users/yzerlaut/work/common_libraries/')
from graphs.my_graph import set_plot, get_linear_colormap, build_bar_legend, put_list_of_figs_to_svg_fig
import data_analysis.fourier_for_real as rfft
sys.path.append('../')
from theory.analytical_calculus import * # where the core calculus lies

#### ================================================== ##
#### MEAN MODEL PROPERTIES ###############################
#### ================================================== ##

soma, stick, params = np.load('mean_model_larkum.npy')
# common to all plots, the frequency range we will look at [0.1,1000] Hz
dt, tstop = 1.3e-3, 10.
t = np.arange(int(tstop/dt))*dt
f = rfft.time_to_freq(len(t), dt)
f, psd_means, phase_means = np.load('../larkumEtAl2009/data/larkum_imped_data.npy')

def get_input_imped(soma, stick, params):
    # branching properties
    params_for_cable_theory(stick, params) # setting cable membrane constants
    output = get_the_input_impedance_at_soma(f, soma, stick, params)
    psd, phase = np.abs(output)/1e6, (np.angle(output)+np.pi)%(2.*np.pi)-np.pi
    return psd, phase

#### ================================================== ##
#### PLOT  ##############################
#### ================================================== ##

def make_experimental_fig():
    fig, AX = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(bottom=.3, left=.25, wspace=.3)
    
    f, psd_means, phase_means = np.load('../larkumEtAl2009/data/larkum_imped_data.npy')
    phase_means = -phase_means

    AX[0].plot(f, psd_means, color='gray', lw=3, label='Layer V pyr. cell \n Larkum et al. 2009')
    AX[1].plot(f, phase_means, color='gray', lw=3, label='Layer V pyr. cell \n Larkum et al. 2009')

    ## THEN MODEL

    from minimization import single_comp_imped
    try:
        Rm, Cm = np.load('single_comp_fit_larkum.npy')
        psd, phase = single_comp_imped(f, Rm, Cm)
        AX[0].loglog(f, psd, 'k:', lw=2)
        AX[1].semilogx(f, phase, 'k:', lw=2, label='single comp.')
    except IOError:
        print('no single compartment data available')
    
    ### MEAN MODEL
    psd, phase = get_input_imped(soma, stick, params)
    AX[0].loglog(f, psd, 'k-', alpha=.8, lw=3)
    AX[1].semilogx(f[:-1], -phase[:-1], 'k-', alpha=.8, lw=3, label='fitted \"reduced model\"')

    ### ===================== FINAL graph settings

    
    AX[1].legend(frameon=False, prop={'size':'x-small'})

    set_plot(AX[0], xlim=[.08,1000], ylim=[.3, 200],\
           xticks=[1,10,100,1000], yticks=[1,10,100],yticks_labels=['1','10','100'],\
             xlabel='frequency (Hz)',ylabel='modulus ($\mathrm{M}\Omega$)')

    set_plot(AX[1], xlim=[.08,1000], ylim=[-.2,2.3],\
               xticks=[1,10,100,1000], yticks=[0,np.pi/4.,np.pi/2.],\
               yticks_labels=[0,'$\pi/4$', '$\pi/2$'],
               xlabel='frequency (Hz)', ylabel='phase shift (Rd)')

    # fig2, ax = make_fig(np.linspace(0, 1, stick['B']+1)*stick['L'],
    #          stick['D'], xscale=1e-6, yscale=50e-6)
    # fig2.set_size_inches(3, 5, forward=True)

    fig3, AX = plt.subplots(1, 2, figsize=(7, 2.5))
    plt.subplots_adjust(left=.3, bottom=.3)
    #### ================================================== ##
    #### COMPARING IT WITH Transfer Resistance DATA ##########
    #### ================================================== ##
    TFdata = np.load('../LarkumEtAl2009/data/larkum_Tf_Resist_data.npz')
    edges, bins = np.histogram(TFdata['R_transfer'], bins=50, weights=TFdata['Areas'], normed=True)
    AX[0].bar(bins[:-1], edges/edges.max(), width=bins[1]-bins[0], color='gray', edgecolor='gray')
    precision = 1e4
    # params_for_cable_theory(stick, params) # setting cable membrane constants
    TF_model, Area = get_the_transfer_resistance_to_soma(soma, stick, params, precision=precision, with_area=True)
    edges, bins = np.histogram(1e-6*TF_model, bins=precision/5, weights=Area, normed=True)
    # edges = np.concatenate([np.array([0,1]), np.real(edges)])
    # bins = np.concatenate([[bins.min()], bins])
    bins = .5*(bins[:-1]+bins[1:])
    AX[0].plot(bins, edges/edges.max(), color='k', lw=1)
    set_plot(AX[0], ylabel='surfacic density \n (norm. by maximum)', xlabel='transfer resistance M$\Omega$', yticks=[0,1])

    IRdata = np.load('../LarkumEtAl2009/data/larkum_Input_Resist_data.npz')
    edges, bins = np.histogram(IRdata['R_input'], bins=50, weights=IRdata['Areas'], normed=True)
    AX[1].bar(bins[:-1], edges/edges.max(), width=bins[1]-bins[0], color='gray', edgecolor='gray')
    precision = 1e4
    # params_for_cable_theory(stick, params) # setting cable membrane constants
    TF_model, Area = get_the_input_resistance(soma, stick, params, precision=precision, with_area=True)
    edges, bins = np.histogram(1e-6*TF_model, bins=precision/10, weights=Area, normed=True)
    # edges = np.concatenate([np.array([0,1]), np.real(edges)])
    # bins = np.concatenate([[bins.min()], bins])
    bins = .5*(bins[:-1]+bins[1:])
    AX[1].plot(bins, edges/edges.max(), color='k', lw=1)
    set_plot(AX[1], ylabel='surfacic density \n (norm. by maximum)', xlabel='input resistance M$\Omega$', yticks=[0,1])
    
    return fig, fig3

if __name__=='__main__':

    from theory.brt_drawing import make_fig # where the core calculus lies
    
    # fig, fig2, fig3 = make_experimental_fig()
    fig, fig3 = make_experimental_fig()
    plt.show()
    # put_list_of_figs_to_svg_fig([fig, fig2], visualize=False)
