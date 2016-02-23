import numpy as np
import sys
sys.path.append('../firing_response_description/')

from excitability_and_sensitivities import get_mean_encoding_power
from template_and_fitting import fitting_Vthre_then_Fout

def produce_reduced_data():
    
    CELLS = []

    OUTPUT = np.zeros((8, 30))
    for i in range(1,31):
        data = np.load('../data_firing_response/cell'+str(i)+'.npz')

        P = fitting_Vthre_then_Fout(data['Fout'], 1e-3*data['muV'],\
                                    1e-3*data['sV'], data['TvN'],\
                                    data['muGn'], data['Gl'], data['Cm'],
                                    data['El'], print_things=True)

        E = get_mean_encoding_power(P, data['El'], data['Gl'], data['Cm'])
        
        CELLS.append({'Gl':data['Gl'], 'Cm':data['Cm'],\
                      'Tm':data['Cm']/data['Gl'], 'P':P, 'E':E})
        OUTPUT[:4,i-1] = P
        OUTPUT[4:,i-1] =E
        print data['Gl'], data['Cm']

    np.save('reduced_data.npy', CELLS)

    return OUTPUT
    
if __name__=='__main__':

    import matplotlib.pylab as plt
    sys.path.append('../code')
    import my_graph as graph
    
    OUTPUT = produce_reduced_data()
    fig, AX = plt.subplots(1, 8, figsize=(15,3))
    # plt.subplots_adjust(bottom=.3)
    for i in range(8):
        AX[i].hist(OUTPUT[i,:])
        graph.set_plot(AX[i], ylabel='cell #')
    if not sys.argv[-1]=='noshow':
        plt.show()

    
