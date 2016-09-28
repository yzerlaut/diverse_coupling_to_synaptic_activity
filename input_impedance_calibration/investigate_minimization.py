import numpy as np
import matplotlib.pylab as plt
import os, sys
sys.path.append('/Users/yzerlaut/work/common_libraries/')
from graphs.my_graph import set_plot, get_linear_colormap, build_bar_legend, put_list_of_figs_to_svg_fig
import data_analysis.fourier_for_real as rfft
sys.path.append('../')
from theory.analytical_calculus import * # where the core calculus lies

all_freq, all_psd, all_phase = np.load('full_data.npy')

# compute mean

f_bins = np.logspace(np.log(.1)/np.log(10), np.log(all_freq.max())/np.log(10), 20)
digitized = np.digitize(all_freq, f_bins)
psd_means = np.array([all_psd[digitized == i].mean() for i in range(1, len(f_bins))])
psd_std = np.array([all_psd[digitized == i].std() for i in range(1, len(f_bins))])
phase_means = np.array([all_phase[digitized == i].mean() for i in range(1, len(f_bins))])
phase_std = np.array([all_phase[digitized == i].std() for i in range(1, len(f_bins))])

f = .5*(f_bins[1:]+f_bins[:-1]) # frequency for theory !!
#### ================================================== ##
#### MEAN MODEL PROPERTIES ###############################
#### ================================================== ##

N = 5
B = np.arange(N)+2 # to be adjusted !!! (does not depends on N)
L_soma = np.linspace(5., 20., N)*1e-6
L_dend = np.linspace(300., 800., N)*1e-6
D_dend = np.linspace(.5, 4., N)*1e-6
G_PAS = np.linspace(1e-5, 1e-4, N)*1e4
CM = np.linspace(.8, 1.8, N)*1e-2
RA = np.linspace(10., 90., N)*1e-2

soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')
try:
    kept_cells = np.load('../coupling_model/kept_cells.npy')
except IOError:
    kept_cells = [True for i in range(30)]


def get_input_imped(soma, stick, params):
    # branching properties
    params_for_cable_theory(stick, params) # setting cable membrane constants
    output = get_the_input_impedance_at_soma(f, soma, stick, params)
    psd, phase = np.abs(output)/1e6, (np.angle(output)+3.*np.pi/2.)%(2.*np.pi)-3.*np.pi/2.
    return psd, -phase

#### ================================================== ##
#### COMPARING IT WITH DATA ##############################
#### ================================================== ##

def make_experimental_fig(index):
    
    fig, AX = plt.subplots(1, 2, figsize=(11,4))
    plt.subplots_adjust(bottom=.2)
    all_freq, all_psd, all_phase = np.load('full_data.npy')

    mymap = get_linear_colormap()
    X = 300+np.arange(3)*300

    HIGH_BOUND = 450 # higher nound for frequency
    f_bins = np.logspace(np.log(.1)/np.log(10), np.log(HIGH_BOUND)/np.log(10), 20)
    digitized = np.digitize(all_freq, f_bins)
    psd_means = np.array([all_psd[digitized == i].mean() for i in range(1, len(f_bins))])
    psd_std = np.array([all_psd[digitized == i].std() for i in range(1, len(f_bins))])
    phase_means = np.array([all_phase[digitized == i].mean() for i in range(1, len(f_bins))])
    phase_std = np.array([all_phase[digitized == i].std() for i in range(1, len(f_bins))])
    AX[0].errorbar(.5*(f_bins[1:]+f_bins[:-1]), psd_means, yerr=psd_std, color='gray', lw=3, label='data')
    AX[0].fill_between(.5*(f_bins[1:]+f_bins[:-1]), psd_means-psd_std, psd_means+psd_std, color='lightgray')
    AX[1].errorbar(.5*(f_bins[1:]+f_bins[:-1]), phase_means, yerr=phase_std, color='gray', lw=3, label='data')
    AX[1].fill_between(.5*(f_bins[1:]+f_bins[:-1]), phase_means-phase_std, phase_means+phase_std, color='lightgray')

    ### MEAN MODEL
    j=0
    for b, ls, ld, dd, g_pas, cm, ra in itertools.product(B, L_soma, L_dend, D_dend, G_PAS, CM, RA):
        if j==index:
            soma1, stick1, params1 = soma.copy(), stick.copy(), params.copy()
            soma1['L'] = ls
            stick1['B'], stick1['D'], stick1['L'] = b, dd, ld
            params1['g_pas'], params1['cm'], params1['Ra'] = g_pas, cm, ra
            psd, phase = get_input_imped(soma1, stick1, params1)
            fig.suptitle('B='+str(b)+', Ls='+str(1e6*ls)+'um, Ld='+str(1e6*ld)+'um, Dd='+\
                         str(1e6*dd)+'um, Gl='+str(1e2*g_pas)+'uS/cm2, Cm='+str(1e2*cm)+'uF/cm2, Ri='+str(1e2*ra)+'Ohm.m')
        j+=1

    AX[0].loglog(f, psd, 'k-', alpha=.8, lw=4)
    AX[1].semilogx(f, phase, 'k-', alpha=.8, lw=4, label='medium size \n   model')
    
    set_plot(AX[0], xlim=[.08,1000],\
                   xticks=[1,10,100,1000], yticks=[10,100,1000],yticks_labels=['10','100','1000'],\
                      xlabel='frequency (Hz)',ylabel='modulus ($\mathrm{M}\Omega$)')

    set_plot(AX[1], xlim=[.08,1000], ylim=[-.2,2.3],\
                   xticks=[1,10,100,1000], yticks=[0,np.pi/4.,np.pi/2.],\
                   yticks_labels=[0,'$\pi/4$', '$\pi/2$'],
             xlabel='frequency (Hz)', ylabel='phase shift (Rd)')

    return fig

import itertools
def compute_deviations(factor_for_phase=3.):
    VALUE_PSD, VALUE_PHASE = [], []
    # product loop
    for b, ls, ld, dd, g_pas, cm, ra in itertools.product(B, L_soma, L_dend, D_dend, G_PAS, CM, RA):
        soma1, stick1, params1 = soma.copy(), stick.copy(), params.copy()
        soma1['L'] = ls
        stick1['B'], stick1['D'], stick1['L'] = b, dd, ld
        params1['g_pas'], params1['cm'], params1['Ra'] = g_pas, cm, ra
        psd, phase = get_input_imped(soma1, stick1, params1)
        VALUE_PSD.append(np.sum((np.log(psd)/np.log(10)-np.log(psd_means)/np.log(10))**2))
        VALUE_PHASE.append(factor_for_phase*np.sum((phase[:-3]-phase_means[:-3])**2))
    np.save('data_minim.npy', [np.array(VALUE_PSD), np.array(VALUE_PHASE)])

def find_minimum_with_distance():
    VALUE_PSD, VALUE_PHASE = np.load('data_minim.npy')
    ERROR = VALUE_PSD/VALUE_PSD.mean()*VALUE_PHASE/VALUE_PHASE.mean()
    i = np.argmin(ERROR)
    imax = np.argmax(ERROR)
    # product loop
    j=0
    for b, ls, ld, dd, g_pas, cm, ra in itertools.product(B, L_soma, L_dend, D_dend, G_PAS, CM, RA):
        if j==i:
            print('====> Minimum of the PHASE and PSD')
            soma1, stick1, params1 = soma.copy(), stick.copy(), params.copy()
            soma1['L'] = ls
            stick1['B'], stick1['D'], stick1['L'] = b, dd, ld
            params1['g_pas'], params1['cm'], params1['Ra'] = g_pas, cm, ra
        if j==imax:
            print('====> Max the PHASE and PSD')
            soma1, stick1, params1 = soma.copy(), stick.copy(), params.copy()
            soma1['L'] = ls
            stick1['B'], stick1['D'], stick1['L'] = b, dd, ld
            params1['g_pas'], params1['cm'], params1['Ra'] = g_pas, cm, ra
        j+=1
    DISTANCE = []
    for b, ls, ld, dd, g_pas, cm, ra in itertools.product(B, L_soma, L_dend, D_dend, G_PAS, CM, RA):
        dist = (b-stick1['B'])/(B[-1]-B[0])*\
               (dd-stick1['D'])/(D_dend[-1]-D_dend[0])*\
               (ld-stick1['L'])/(L_dend[-1]-L_dend[0])*\
               (g_pas-params1['g_pas'])/(G_PAS[-1]-G_PAS[0])*\
               (cm-params1['cm'])/(CM[-1]-CM[0])*\
               (ra-params1['Ra'])/(RA[-1]-RA[0])
        DISTANCE.append(dist)
    return np.array(DISTANCE), np.array(ERROR)

def plot_traces_of_intervals(I1=[1e-9,0.002], I2=[0.01,0.1], I3=[0.1,1.], I4=[1,10.], number=2):
    VALUE_PSD, VALUE_PHASE = np.load('data_minim.npy')
    ERROR = VALUE_PSD/VALUE_PSD.mean()*VALUE_PHASE/VALUE_PHASE.mean()

    FIGS = []
    for I in [I1, I2, I3, I4]:
        sample = np.argwhere((ERROR>I[0]) & (ERROR<I[1]))
        for ii in np.random.randint(len(sample), size=number):
            fig = make_experimental_fig(sample[ii])
            FIGS.append(fig)
    return FIGS

def make_first_fig():
    DISTANCE, ERROR = find_minimum_with_distance()
    N = 40
    bins_positive = np.logspace(-9,np.log(DISTANCE.max())/np.log(10),N)
    bins_negative = np.logspace(-9,np.log(-DISTANCE.min())/np.log(10),N)
    i_positive = np.digitize(DISTANCE[DISTANCE>0], bins_positive)
    i_negative = np.digitize(-DISTANCE[DISTANCE<0], bins_negative)

    x_plus, val_plus, val_min = [], [], []
    x_min, val_plus_std, val_min_std = [], [], []
    for i in range(N):
        if len(ERROR[DISTANCE>0][i_positive==i])>0:
            val_plus.append(np.mean(ERROR[DISTANCE>0][i_positive==i]))
            val_plus_std.append(np.std(ERROR[DISTANCE>0][i_positive==i]))
            x_plus.append(bins_positive[i])
        if len(ERROR[DISTANCE>0][i_positive==i])>0:
            val_min.append(np.mean(ERROR[DISTANCE<0][i_negative==i]))
            val_min_std.append(np.std(ERROR[DISTANCE<0][i_negative==i]))
            x_min.append(bins_negative[i])

    sys.path.append('../../')
    from graphs.my_graph import set_plot
    fig, AX = plt.subplots(1,2, figsize=(10,4))
    AX[1].plot([DISTANCE[DISTANCE>0].min(),DISTANCE.max()],\
               [ERROR.min(),ERROR.min()], '-', color='lightgray', lw=5)
    AX[1].plot(DISTANCE[DISTANCE>0], ERROR[DISTANCE>0], 'k.', alpha=.05)
    AX[1].errorbar(x_plus, val_plus, yerr=val_plus_std, color='k', lw=3)
    AX[1].set_xscale('log');AX[1].set_yscale('log')
    set_plot(AX[1], yticks=np.logspace(-2,0,3), xticks=np.logspace(-2,0,3), ylabel='error (log)')
    
    AX[0].plot([DISTANCE[DISTANCE>0].min(),DISTANCE.max()],\
               [ERROR.min(),ERROR.min()], '-', color='lightgray', lw=5)
    AX[0].plot(-DISTANCE[DISTANCE<0], ERROR[DISTANCE<0], 'k.', alpha=.05)
    AX[0].errorbar(x_min, val_min, yerr=val_min_std, color='k', lw=3)
    AX[0].set_xscale('log');AX[0].set_yscale('log')
    set_plot(AX[0], yticks=np.logspace(-2,0,3), xticks=np.logspace(-2,0,3), ylabel='error (log)')

    return fig

def make_second_fig():
    DISTANCE, ERROR = find_minimum_with_distance()
    bins = np.logspace(np.log(ERROR.min())/np.log(10),np.log(ERROR.max())/np.log(10),10)
    hist, be = np.histogram(ERROR, bins=bins)
    
    from graphs.my_graph import set_plot
    fig, AX = plt.subplots(1, figsize=(5,4))
    plt.subplots_adjust(bottom=.3, left=.2)
    AX.bar(be[:-1], hist, width=np.diff(bins), edgecolor='k', color='lightgray', lw=2)
    AX.set_xscale('log')
    set_plot(AX, xticks=np.logspace(-2,1,4), xlabel='goodness to fit', ylabel='morphology number')
    return fig

if __name__=='__main__':

    if sys.argv[-1]=='compute':
        compute_deviations()
    elif sys.argv[-1]=='distance':
        fig = make_first_fig()
    elif sys.argv[-1]=='hist':
        fig = make_second_fig()
        fig.savefig('fig.svg')
    else:
        FIGS = plot_traces_of_intervals()
        ii=0
        for fig in FIGS:
            fig.savefig('fig'+str(ii)+'.svg')
            ii+=1
    # plt.show()
