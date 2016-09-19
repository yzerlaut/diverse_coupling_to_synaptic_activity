import numpy as np
import matplotlib.pylab as plt
import os, sys
sys.path.append('../code')
import my_graph as graph
sys.path.append('../')
from theory.analytical_calculus import * # where the core calculus lies
from membrane_parameters import soma, stick, params # BASELINE PARAMETERS

#### ================================================== ##
#### LOAD DATA ###############################
#### ================================================== ##

f, psd_means, phase_means = np.load('../larkumEtAl2009/data/larkum_imped_data.npy')
phase_means = -phase_means

#### ================================================== ##
#### ####### MODEL VARIATIONS ############################
#### ================================================== ##

N0, N = 5, 5
B = np.arange(N)+N0 # to be adjusted !!! (does not depends on N)
L_soma = np.linspace(5., 60., N)*1e-6
L_dend = np.linspace(700., 1200., N)*1e-6
D_dend = np.linspace(3., 7., N)*1e-6
G_PAS = np.linspace(17e-5, 3e-5, N)*1e4
CM = np.linspace(.8, 1.8, N)*1e-2
RA = np.linspace(10., 90., N)*1e-2

#### ================================================== ##
#### MODEL PROPERTIES ###############################
#### ================================================== ##


def get_input_imped(soma, stick, params):
    # branching properties
    params_for_cable_theory(stick, params) # setting cable membrane constants
    output = get_the_input_impedance_at_soma(f, soma, stick, params)
    psd, phase = np.abs(output)/1e6, (np.angle(output)+np.pi/2.)%(2.*np.pi)-np.pi/2.
    return psd, -phase

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
        VALUE_PHASE.append(factor_for_phase*np.sum((phase-phase_means)**2))
    np.save('data_minim_larkum.npy', [np.array(VALUE_PSD), np.array(VALUE_PHASE)])

def find_minimum():
    try :
        VALUE_PSD, VALUE_PHASE = np.load('data_minim_larkum.npy')
        i0 = np.argmin(VALUE_PSD)
        i1 = np.argmin(VALUE_PHASE)
        i = np.argmin(VALUE_PSD/VALUE_PSD.mean()*VALUE_PHASE/VALUE_PHASE.mean())
        # product loop
        j=0
        for b, ls, ld, dd, g_pas, cm, ra in itertools.product(B, L_soma, L_dend, D_dend, G_PAS, CM, RA):
            if j==i:
                print('====> Minimum of the PHASE and PSD')
                soma1, stick1, params1 = soma.copy(), stick.copy(), params.copy()
                soma1['L'] = ls
                stick1['B'], stick1['D'], stick1['L'] = b, dd, ld
                params1['g_pas'], params1['cm'], params1['Ra'] = g_pas, cm, ra
                print('B=',b, ', Ls=',1e6*ls, 'um, Ld=',1e6*ld, 'um, Dd=',\
                          1e6*dd,'um,\n g_pas=',g_pas,', cm=', 1e2*cm, ', Ra=',1e2*ra)
                MIN_BOTH = [soma1.copy(), stick1.copy(), params1.copy()]
            if j==i0:
                print('====> Minimum of the PSD')
                soma1, stick1, params1 = soma.copy(), stick.copy(), params.copy()
                soma1['L'] = ls
                stick1['B'], stick1['D'], stick1['L'] = b, dd, ld
                params1['g_pas'], params1['cm'], params1['Ra'] = g_pas, cm, ra
                print('B=',b, ', Ls=',1e6*ls, 'um, Ld=',1e6*ld, 'um, Dd=',\
                          1e6*dd,'um,\n g_pas=',g_pas,', cm=', 1e2*cm, ', Ra=',1e2*ra)
                MIN_PSD = [soma1.copy(), stick1.copy(), params1.copy()]
            if j==i1:
                print('====> Minimum of the PHASE')
                soma1, stick1, params1 = soma.copy(), stick.copy(), params.copy()
                soma1['L'] = ls
                stick1['B'], stick1['D'], stick1['L'] = b, dd, ld
                params1['g_pas'], params1['cm'], params1['Ra'] = g_pas, cm, ra
                print('B=',b, ', Ls=',1e6*ls, 'um, Ld=',1e6*ld, 'um, Dd=',\
                          1e6*dd,'um,\n g_pas=',g_pas,', cm=', 1e2*cm, ', Ra=',1e2*ra)
                MIN_PHASE = [soma1.copy(), stick1.copy(), params1.copy()]
            j+=1
        return MIN_PHASE, MIN_PSD, MIN_BOTH
    except IOError:
        print('--------------------------------------------')
        print('NEED TO MAKE THE MINIMIZATION FIRST')
        print('RUN :')
        print('python minimization compute')
        print('--------------------------------------------')

def make_fig(MIN_PHASE, MIN_PSD, MIN_BOTH):
    fig, AX = plt.subplots(1, 2, figsize=(11,4))

    AX[0].plot(f, psd_means, color='gray', lw=3, label='Layer V pyr. cell')
    AX[1].plot(f, phase_means, color='gray', lw=3, label='Layer V pyr. cell')

    for p, label in zip([MIN_PHASE, MIN_PSD, MIN_BOTH],\
                        ['phase min.', 'psd min.', 'both min.']):
        soma1, stick1, params1 = p
        psd, phase = get_input_imped(soma1, stick1, params1)
        AX[0].loglog(f, psd, label=label, lw=2)
        AX[1].semilogx(f, phase, label=label, lw=2)
    AX[0].legend(prop={'size':'xx-small'}, loc='best')
    AX[1].legend(prop={'size':'xx-small'}, loc='best')


def single_comp_imped(f, Rm, Cm):
    imped = Rm/(1.+1j*2.*np.pi*f*Rm*Cm)
    return np.abs(imped), -np.angle(imped)

def single_comp_minim():
    
    from scipy.optimize import minimize

    def Res(p):
        psd, phase = single_comp_imped(f, p[0], p[1])
        return np.sum((psd-psd_means)**2)*np.sum((phase+phase_means)**2)

    P0 = [400, 50e-6]
    psd, phase = single_comp_imped(f, P0[0], P0[1])
    print(phase_means[-1], phase[-1])
    plsq = minimize(Res,P0, method='nelder-mead')

    np.save('single_comp_fit_larkum.npy', plsq.x)
    print(plsq)

if __name__=='__main__':

    if sys.argv[-1]=='compute':
        compute_deviations()
    elif sys.argv[-1]=='single_comp':
        single_comp_minim() 
    else:
        MIN_PHASE, MIN_PSD, MIN_BOTH = find_minimum()
        make_fig(MIN_PHASE, MIN_PSD, MIN_BOTH)
        if not sys.argv[-1]=='noshow':
            plt.show()
        np.save('mean_model_larkum.npy', MIN_BOTH)



