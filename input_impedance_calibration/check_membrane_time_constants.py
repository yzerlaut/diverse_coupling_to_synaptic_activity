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

f=rfft.time_to_freq(1000, 1e-4)

# loading mean model
soma, stick, params = np.load('../input_impedance_calibration/mean_model.npy')
EqCylinder = np.linspace(0, 1, stick['B']+1)*stick['L']

CELLS = np.load('../data_firing_response/reduced_data.npy')
Tm_data, Tm_model = np.zeros(len(CELLS)), np.zeros(len(CELLS))

for i in range(len(CELLS)):
    Rm = 1e-6/CELLS[i]['Gl']
    soma1, stick1, params1 = adjust_model_prop(Rm, soma, stick)
    
    EqCylinder1 = np.linspace(0, 1, stick1['B']+1)*stick1['L']
    psd = np.abs(get_the_input_impedance_at_soma(f, EqCylinder1, soma1, stick1, params1))**2

    Tm_model[i]= .5*psd[0]/(2.*np.trapz(np.abs(psd), f)) # 2 times the integral to have from -infty to +infty (and methods gives [0,+infty])
    Tm_data[i]= CELLS[i]['Cm']/CELLS[i]['Gl']
    

print 'Comparison between model and data for Tm'
print 'DATA, mean = ', 1e3*Tm_data.mean(), 'ms +/-', 1e3*Tm_data.std()
print 'MODEL, mean = ', 1e3*Tm_model.mean(), 'ms +/-', 1e3*Tm_model.std()

fig, ax = plt.subplots(figsize=(4,3))
plt.subplots_adjust(bottom=.3, left=.25)
plt.hist(1e3*Tm_data, color='r', label='data')
plt.hist(1e3*Tm_model, color='b', label='model')
plt.legend(frameon=False, prop={'size':'x-small'}, loc='best')
set_plot(ax, xlabel='membrane time constant (ms)', ylabel='cell #')
plt.show()
