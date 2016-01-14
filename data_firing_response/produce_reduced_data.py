import numpy as np
import sys
sys.path.append('../')

from firing_response_description.template_and_fitting import fitting_Vthre_then_Fout

def produce_reduced_data():
    
    CELLS = []

    for i in range(1,31):
        data = np.load('../data_firing_response/cell'+str(i)+'.npz')

        P = fitting_Vthre_then_Fout(data['Fout'], 1e-3*data['muV'],\
                                    1e-3*data['sV'], data['TvN'],\
                                    data['muGn'], data['Gl'], data['Cm'],
                                    data['El'], print_things=True)
        CELLS.append({'Gl':data['Gl'], 'Cm':data['Cm'],\
                      'Tm':data['Cm']/data['Gl'], 'P':P})
        print data['Gl'], data['Cm']

    np.save('reduced_data.npy', CELLS)
    
if __name__=='__main__':
    produce_reduced_data()
