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

soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')
try:
    kept_cells = np.load('../coupling_model/kept_cells.npy')
except IOError:
    kept_cells = [True for i in range(30)]


# common to all plots, the frequency range we will look at [0.1,1000] Hz
dt, tstop = 1.3e-3, 10.
t = np.arange(int(tstop/dt))*dt
f = rfft.time_to_freq(len(t), dt)

def get_input_imped(soma, stick, params):
    # branching properties
    params_for_cable_theory(stick, params) # setting cable membrane constants
    output = get_the_input_impedance_at_soma(f, soma, stick, params)
    psd, phase = np.abs(output)/1e6, (np.angle(output)+np.pi)%(2.*np.pi)-np.pi
    return psd, phase

def get_input_resist(soma, stick, params):
    # branching properties
    params_for_cable_theory(stick, params) # setting cable membrane constants
    return np.abs(get_the_input_impedance_at_soma(0., soma, stick, params))

def adjust_model_prop(Rm, soma, stick, precision=.5, params2=None, maxiter=1000):
    """ Rm in Mohm !! """
    if Rm>1200 or Rm<100:
        print('---------------------------------------------------')
        print('/!\ Rm value too high or too low for the conversion')
        print('---------------------------------------------------')

    # comodulation values :
    L_soma_base = 2e-6
    L_dend_base = 150e-6
    D_dend_base = .8*1e-6

    def get_Rm_from_factor(b):
        soma1, stick1 = soma.copy(), stick.copy()
        if params2 is None:
            params1 = params.copy()
        else:
            params1=params2
        soma1['L'] += b*L_soma_base
        stick1['L'] += b*L_dend_base
        stick1['D'] += b*D_dend_base
        return get_input_resist(soma1, stick1, params1)
    
    b, delta_b = 0., .3
    n, n_same = 0, 0
    diff = Rm-1e-6*get_Rm_from_factor(b)
    previous_diff = diff
    while abs(diff)>precision and n<maxiter:
        diff = Rm-1e-6*get_Rm_from_factor(b)

        if diff*previous_diff<0: # if we jumped over the value
            delta_b /=2. # need to make the bin smaller
            n_same = 0 # we changed side
        else:
            n_same +=1
            if n_same>20:
                delta_b = 0.1
            
        if diff>0: # if model cell too big, but we got closer
            b -= delta_b # we make the same jump than before
        elif diff<0: # if model cell too big, but we got closer
            b += delta_b # we make the same jump than before

        previous_b = b
        previous_diff = diff
        n+=1
    if n==maxiter:
        print(delta_b)
        print(n_same)
        print('minimization not achieved !!!')
        print(Rm, 1e-6*get_Rm_from_factor(b))
        
    if params2 is None:
        params1 = params.copy()
    else:
        params1=params2
    soma1, stick1 = soma.copy(), stick.copy()
    soma1['L'] += b*L_soma_base
    stick1['L'] += b*L_dend_base
    stick1['D'] += b*D_dend_base
    return soma1.copy(), stick1.copy(), params1.copy()

#### ================================================== ##
#### COMPARING IT WITH DATA ##############################
#### ================================================== ##

def make_experimental_fig():
    fig, AX = plt.subplots(2, 2, figsize=(11,8))
    MICE, RATS = [], []

    psd_boundaries = [200,900]
    all_freq, all_psd, all_phase = [np.empty(0, dtype=float) for i in range(3)]
    HIGH_BOUND = 450 # higher nound for frequency

    for file in os.listdir("./intracellular_data/"):
        if file.endswith("_rat.txt"):
            _ = 0
        elif file.endswith(".txt"):
            freq, psd, phase = np.loadtxt("./intracellular_data/"+file, unpack=True)
            MICE.append({'freq':freq[np.argsort(freq)], 'psd':psd[np.argsort(freq)], 'phase':(phase[np.argsort(freq)]%(2.*np.pi)+np.pi/2.)%(2.*np.pi)-np.pi/2.})
            # print file, MICE[-1]['phase'].max()
            all_freq = np.concatenate([all_freq, MICE[-1]['freq']])
            all_psd = np.concatenate([all_psd, MICE[-1]['psd']])
            all_phase = np.concatenate([all_phase, MICE[-1]['phase']])

        psd_boundaries[0] = min([psd_boundaries[0], psd.max()])
        psd_boundaries[1] = max([psd_boundaries[1], psd.max()])

    np.save('full_data.npy',\
        [all_freq[all_freq<HIGH_BOUND], all_psd[all_freq<HIGH_BOUND], all_phase[all_freq<HIGH_BOUND]])
    all_freq, all_psd, all_phase = np.load('full_data.npy')

    mymap = get_linear_colormap()
    X = 300+np.arange(3)*300

    for m in MICE:
        rm = m['psd'][:3].mean()
        r = (rm-psd_boundaries[0])/(psd_boundaries[1]-psd_boundaries[0])
        AX[0,0].loglog(m['freq'][m['freq']<HIGH_BOUND], m['psd'][m['freq']<HIGH_BOUND], 'o-', color=mymap(r,1), ms=3)
        AX[0,1].semilogx(m['freq'][m['freq']<HIGH_BOUND], m['phase'][m['freq']<HIGH_BOUND], 'o-', color=mymap(r,1), ms=3)

    AX[0,0].annotate('n='+str(len(MICE))+' cells', (.2,.2), xycoords='axes fraction', fontsize=18)

    ax2 = fig.add_axes([0.59,0.7,0.015,0.18])
    cb = build_bar_legend(X, ax2, mymap, scale='linear', color_discretization=100,\
                                bounds=psd_boundaries,
                                label=r'$R_\mathrm{m}$ ($\mathrm{M}\Omega$)')

    f_bins = np.logspace(np.log(.1)/np.log(10), np.log(HIGH_BOUND)/np.log(10), 20)
    digitized = np.digitize(all_freq, f_bins)
    psd_means = np.array([all_psd[digitized == i].mean() for i in range(1, len(f_bins))])
    psd_std = np.array([all_psd[digitized == i].std() for i in range(1, len(f_bins))])
    phase_means = np.array([all_phase[digitized == i].mean() for i in range(1, len(f_bins))])
    phase_std = np.array([all_phase[digitized == i].std() for i in range(1, len(f_bins))])
    AX[1,0].errorbar(.5*(f_bins[1:]+f_bins[:-1]), psd_means, yerr=psd_std, color='gray', lw=3, label='data')
    AX[1,0].fill_between(.5*(f_bins[1:]+f_bins[:-1]), psd_means-psd_std, psd_means+psd_std, color='lightgray')
    AX[1,1].errorbar(.5*(f_bins[1:]+f_bins[:-1]), phase_means, yerr=phase_std, color='gray', lw=3, label='data')
    AX[1,1].fill_between(.5*(f_bins[1:]+f_bins[:-1]), phase_means-phase_std, phase_means+phase_std, color='lightgray')

    ## THEN MODEL

    from minimization import single_comp_imped
    try:
        Rm, Cm = np.load('single_comp_fit.npy')
        psd, phase = single_comp_imped(f, Rm, Cm)
        AX[1,0].loglog(f, psd, 'k:', lw=2)
        AX[1,1].semilogx(f, phase, 'k:', lw=2, label='single comp.')
    except IOError:
        print('no single compartment data available')
    
    ### MEAN MODEL
    psd, phase = get_input_imped(soma, stick, params)
    AX[1,0].loglog(f, psd, 'k-', alpha=.8, lw=4)
    AX[1,1].semilogx(f[:-1], -phase[:-1], 'k-', alpha=.8, lw=4, label='medium size \n   model')

    
    ### MODEL VARIATIONS
    base = np.linspace(0,1,4)
    for Rrm, b in zip([250, 350, 600, 800], base):
        soma1, stick1, params1 = adjust_model_prop(Rrm, soma, stick)
        psd, phase = get_input_imped(soma1, stick1, params1)
        AX[1,0].loglog(f, psd, '-', color=mymap(b,1), ms=5)
        AX[1,1].semilogx(f[:-1], -phase[:-1], '-', color=mymap(b,1), ms=5)

    ### ===================== FINAL graph settings


    ax3 = fig.add_axes([0.19,0.125,0.015,0.18])
    cb = build_bar_legend([0,1], ax3, mymap, scale='linear', color_discretization=100,\
                                bounds=psd_boundaries,ticks_labels=['',''],
                                label='size variations')
    
    AX[1,1].legend(frameon=False, prop={'size':'x-small'})

    for ax, xlabel in zip([AX[0,0], AX[1,0]], ['','frequency (Hz)']):
        set_plot(ax, xlim=[.08,1000], ylim=[6., 1200],\
                   xticks=[1,10,100,1000], yticks=[10,100,1000],yticks_labels=['10','100','1000'],\
                       xlabel=xlabel,ylabel='modulus ($\mathrm{M}\Omega$)')

    for ax, xlabel in zip([AX[0,1], AX[1,1]], ['','frequency (Hz)']):
        set_plot(ax, xlim=[.08,1000], ylim=[-.2,2.3],\
                   xticks=[1,10,100,1000], yticks=[0,np.pi/4.,np.pi/2.],\
                   yticks_labels=[0,'$\pi/4$', '$\pi/2$'],
                   xlabel=xlabel, ylabel='phase shift (Rd)')

    fig2, ax = make_fig(np.linspace(0, 1, stick['B']+1)*stick['L'],
             stick['D'], xscale=1e-6, yscale=50e-6)
    fig2.set_size_inches(3, 5, forward=True)

    fig3 = make_conversion_fig(Rm0=Rm)
    
    return fig, fig2, fig3

from data_firing_response.analyze_data import get_Rm_range
def make_conversion_fig(Rm0=None):
        # we plot here the conversion between input resistance
        # and dendritic tree properties
        Rm_exp = get_Rm_range()[kept_cells]
        Rm = np.linspace(0.9*Rm_exp.min(), 1.2*Rm_exp.max(), 20)
        LS, LD, DD = 0*Rm, 0*Rm, 0*Rm

        for i in range(len(Rm)):
            soma1, stick1, params1 = adjust_model_prop(Rm[i], soma, stick,\
                                                       precision=.01, maxiter=1e5)
            LS[i] = 1e6*soma1['L']
            DD[i], LD[i] = 1e6*stick1['D'], 1e6*stick1['L']

        fig, AX = plt.subplots(4, figsize=(4.5,8))
        plt.subplots_adjust(left=.4, bottom=.15, hspace=.3)
        
        AX[0].set_title('resizing rule')
        AX[0].hist(Rm_exp, bins=np.linspace(Rm.min(), Rm.max(), 10),\
                   color='lightgray', edgecolor='k', lw=2)
        set_plot(AX[0], ['left'], ylabel='cell #', xticks=[])
                   
        for ax, y, label in zip(AX[1:3], [LS, DD],\
             ['length of \n soma ($\mu$m)', 'root branch \n diameter ($\mu$m)']):
            ax.plot(Rm, y, 'k-', lw=2)
            set_plot(ax, ['left'], ylabel=label, xticks=[])
        AX[3].plot(Rm, LD, 'k-', lw=2)
        set_plot(AX[3], ylabel='tree length \n ($\mu$m)',\
                       xlabel='somatic input \n resistance (M$\Omega$)',\
                       xticks=[200,400,600])

        soma0, stick0, params0 = adjust_model_prop(Rm_exp.mean(),\
                                                   soma, stick, precision=.1, maxiter=1e4)
        if Rm0 is not None:
            AX[1].plot([Rm_exp.mean()], [1e6*soma0['L']], 'kD')
            AX[2].plot([Rm_exp.mean()], [1e6*stick0['D']], 'kD')
            AX[3].plot([Rm_exp.mean()], [1e6*stick0['L']], 'kD')
            
        return fig

if __name__=='__main__':

    from theory.brt_drawing import make_fig # where the core calculus lies
    
    fig, fig2, fig3 = make_experimental_fig()
    plt.show()
    put_list_of_figs_to_svg_fig([fig, fig2, fig3], visualize=False)
