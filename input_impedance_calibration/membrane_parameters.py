# baseline somatic parameters, will be modified by geometry
soma = {'L': 10*1e-6, 'D': 15*1e-6, 'NSEG': 1,\
        'exc_density':1e9, 'inh_density':(1e-5)**2/20., 'name':'soma'}

# baseline stick parameters, will be modified by geometry
stick = {'L': 500*1e-6, 'D': 1.*1e-6, 'B':10, 'NSEG': 30,\
         'exc_density':(1e-5)**2/30., 'inh_density':(1e-5)**2/6., 'name':'dend'}

# biophysical properties
params = {'g_pas': 1e-4*1e4, 'cm' : 1.*1e-2, 'Ra' : 200.*1e-2}

# now synaptic properties
params['Qe'], params['Qi'] = 0.7e-9, 1e-9
params['Te'], params['Ti'] = 5e-3, 5e-3
params['Ee'], params['Ei'] = 0e-3, -80e-3
params['El'] = -65e-3#0e-3, -80e-3

params['fraction_for_L_prox'] = 5./6.
params['factor_for_distal_synapses_weight'] = 2.
params['factor_for_distal_synapses_tau'] = 1.

def format_params(soma, stick, params):
    params['rm'] = 10./params['g_pas'] # S/m2 ->
    params['exc_density'] = 10./params['g_pas'] # S/m2 ->
    S1 ="""| Membrane Parameters |
| leak resistivity density | r_m | %(rm)s | mS/cm^2 |
| excitatory synapses density | \mathcal{D_e} | %(exc_density)f | synapses/cm^2 |
| leak resistivity density | r_m | %(rm)f | mS/cm^2 |
    """ %params
    soma['D'] *=1e6
    soma['L'] *=1e6
    S2 ="""
| soma diameter | D_S | %(D)s | $\mu m$ |
| soma length (mean model) | L_S | %(L)s | $\mu m$ |
    """ %soma
    print S1+S2
    
if __name__=='__main__':
    ## | Parameter Name          | Symbol | Value | Unit |
    format_params(soma, stick, params)
