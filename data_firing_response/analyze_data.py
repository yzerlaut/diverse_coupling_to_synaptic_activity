import numpy as np
import matplotlib.pylab as plt


def get_Rm_range(plot=False):
    
    Rm = np.zeros(30)
    for i in range(1,31):
        data = np.load('../data_firing_response/cell'+str(i)+'.npz')
        Rm[i-1] = 1e-6/data['Gl']

    if plot:
        print 'Rm range: ', Rm.min(), Rm.max()
        plt.hist(Rm, bins=10)
        plt.show()

    return Rm


##### FITTING OF THE PHENOMENOLOGICAL THRESHOLD #####
# two-steps procedure, see template_and_fitting.py
# need SI units !!!
# P = fitting_Vthre_then_Fout(data['Fout'], 1e-3*data['muV'],\
#                             1e-3*data['sV'], data['TvN'],\
#                             data['muGn'], data['Gl'], data['Cm'],
#                             data['El'], print_things=True)


if __name__=='__main__':
    get_Rm_range(plot=True)
