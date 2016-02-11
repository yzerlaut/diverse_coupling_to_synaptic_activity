
# baseline somatic parameters, will be modified by geometry
soma = {'L': 10*1e-6, 'D': 15*1e-6, 'NSEG': 1,\
        'exc_density':1e9, 'inh_density':(1e-5)**2/15., 'name':'soma'}

# baseline stick parameters, will be modified by geometry
stick = {'L': 500*1e-6, 'D': 1.*1e-6, 'B':10, 'NSEG': 30,\
         'exc_density':(1e-5)**2/40., 'inh_density':(1e-5)**2/10., 'name':'dend'}

# biophysical properties
params = {'g_pas': 1e-4*1e4, 'cm' : 1.*1e-2, 'Ra' : 200.*1e-2, 'El': -65e-3,
          'Qe' : 1.e-9 , 'Te' : 5.e-3, 'Ee': 0e-3,
          'Qi' : 1.5e-9 , 'Ti' : 5.e-3, 'Ei': -80e-3,
          'Ee': 0e-3, 'Ei': -80e-3}

params['Qe'], params['Qi'] = 1.e-9, 1.2e-9
params['Te'], params['Ti'] = 4e-3, 4e-3
params['Ee'], params['Ei'] = 0e-3, -80e-3
params['El'] = -60e-3#0e-3, -80e-3
params['fraction_for_L_prox'] = 2./3.
params['factor_for_distal_synapses_weight'] = 2.
params['factor_for_distal_synapses_tau'] = 2.

