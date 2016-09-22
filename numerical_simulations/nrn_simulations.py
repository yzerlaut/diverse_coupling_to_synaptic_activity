# nrn_simulations.py
import platform, sys
if platform.system()=='Darwin':
    sys.path.append('/Applications/NEURON-7.4/nrn/lib/python')
    sys.path.append('/System/Library/Frameworks/Python.framework/Versions/2.7')
    sys.path.append('/System/Library/Frameworks/Python.framework/Versions/2.7/lib')
from neuron import h as nrn
nrn('objref nil') # need a nil object in NEURON
import numpy as np
from shotnoise import *

def set_tree_params(EqCylinder, dend, soma, Params):
    """ returns the different diameters of the equivalent cylinder
    given a number of branches point"""
    cables, xtot = [], np.zeros(1)
    cables.append(soma.copy())
    cables[0]['inh_density'] = Params['inh_density_soma']
    cables[0]['exc_density'] = Params['exc_density_soma']
    Ke_tot, Ki_tot = 0, 0
    D = dend['D'] # mothers branch diameter
    for i in range(1,len(EqCylinder)):
        cable = dend.copy()
        cable['x1'], cable['x2'] = EqCylinder[i-1], EqCylinder[i]
        cable['L'] = cable['x2']-cable['x1']
        dx = cable['L']/cable['NSEG']
        cable['x'] = EqCylinder[i-1]+dx/2+np.arange(cable['NSEG'])*dx
        xtot = np.concatenate([xtot, cable['x']])
        cable['D'] = D*2**(-2*(i-1)/3.)
        cable['inh_density'] = Params['inh_density_dend']
        cable['exc_density'] = Params['exc_density_dend']
        cables.append(cable)

    Ke_tot, Ki_tot, jj = 0, 0, 0
    for cable in cables:
        cable['Ki_per_seg'] = cable['L']*\
          cable['D']*np.pi/cable['NSEG']/cable['inh_density']
        cable['Ke_per_seg'] = cable['L']*\
          cable['D']*np.pi/cable['NSEG']/cable['exc_density']
        # summing over duplicate of compartments
        Ki_tot += 2**jj*cable['Ki_per_seg']*cable['NSEG']
        Ke_tot += 2**jj*cable['Ke_per_seg']*cable['NSEG']
        if cable['name']!='soma':
            jj+=1
    print(("Total number of EXCITATORY synapses : ", Ke_tot))
    print(("Total number of INHIBITORY synapses : ", Ki_tot))
    return xtot, cables

def Constructing_the_ball_and_tree(params, cables,
                                   spiking_mech=False, nmda_on=False, Ca_spikes_on=False, HH_on=False):

    # list of excitatory stuff
    exc_synapses, exc_netcons, exc_spike_trains, exc_Ks  = [], [], [], []
    # list of inhibitory stuff
    inh_synapses, inh_netcons, inh_spike_trains, inh_Ks = [], [], [], []
    # area list
    area_lists = []
    
    area_tot = 0

    ## --- CREATING THE NEURON
    for level in range(len(cables)):
        
        # list of excitatory stuff per level
        exc_synapse, exc_netcon, exc_spike_train, exc_K = [], [], [], []
        # list of inhibitory stuff per level
        inh_synapse, inh_netcon, inh_spike_train, inh_K = [], [], [], []
        
        area_list = []

        for comp in range(max(1,2**(level-1))):
                nrn('create cable_'+str(int(level))+'_'+\
                    str(int(comp)))
                exec('section = nrn.cable_'+str(level)+'_'+str(comp))

                ## --- geometric properties
                section.L = cables[level]['L']*1e6
                section.diam = cables[level]['D']*1e6
                section.nseg = cables[level]['NSEG']
                
                ## --- passive properties
                section.Ra = params['Ra']*1e2
                section.cm = params['cm']*1e2
                section.insert('pas')
                section.g_pas = params['g_pas']*1e-4
                section.e_pas = params['El']*1e3
                if comp==0 and HH_on:
	            section.insert('hh3Larkum2009')
                if (comp==1 or comp==2) and Ca_spikes_on:
                    section.insert('scaLarkum2009')
                    section.gbar_scaLarkum2009=10
	            # section.insert('kdrLarkum2009')
                    # section.insert('kcaLarkum2009')
                    # section.gbar_kcaLarkum2009=30
                    section.insert('cad2Larkum2009')
	            section.insert('it2Larkum2009')
                    # section.gcabar_it2Larkum2009=0.000
	            # section.insert('hh3Larkum2009')
                    # section.gkbar_hh3Larkum2009=0
                    # section.gkbar2_hh3Larkum2009=0.0001
                    # section.gnabar_hh3Larkum2009=0.01
	            # section.insert('kapLarkum2009')
                    # section.gkabar_kapLarkum2009=0.003
	            # section.insert('kmLarkum2009')
                    # section.gbar_kmLarkum2009=0
                    # section.gbar_kdrLarkum2009=0

                if level>=1: # if not soma
                    # in NEURON : connect daughter, mother branch
                    nrn('connect cable_'+str(int(level))+'_'+\
                        str(int(comp))+'(0), '+\
                        'cable_'+str(int(level-1))+\
                        '_'+str(int(comp/2.))+'(1)')
                    
                ## --- SPREADING THE SYNAPSES (bis)
                for seg in section:

                    tree_fraction = np.max([0,float(seg.x)/(len(cables)-1)+float(level-1)/(len(cables)-1)])
                    if tree_fraction>params['fraction_for_L_prox']:
                        # print 'strengthen synapse !'
                        Ftau = 1. # removed params !
                        Fq = params['factor_for_distal_synapses_weight']
                    else:
                        Ftau = 1.
                        Fq = 1.

                    # in each segment, we insert an excitatory synapse
                    if nmda_on:
                        syn = nrn.my_glutamate(seg.x, sec=section)
                        syn.tau_ampa, syn.e = Ftau*params['Te']*1e3, params['Ee']*1e3
                        syn.gmax = 1e9*params['Qe'] # nS in my_glutamate.mod
                    else:
                        syn = nrn.ExpSyn(seg.x, sec=section)
                        syn.tau, syn.e = Ftau*params['Te']*1e3, params['Ee']*1e3
                    exc_synapse.append(syn)
                    netcon = nrn.NetCon(nrn.nil, syn)
                    netcon.weight[0] = Fq*params['Qe']*1e6
                    exc_netcon.append(netcon)
                    exc_K.append(cables[level]['Ke_per_seg'])
                    exc_spike_train.append([])

                    syn = nrn.ExpSyn(seg.x, sec=section)
                    syn.tau, syn.e = Ftau*params['Ti']*1e3, params['Ei']*1e3
                    inh_synapse.append(syn)
                    inh_K.append(cables[level]['Ki_per_seg'])
                    inh_spike_train.append([])
                    netcon = nrn.NetCon(nrn.nil, syn)
                    netcon.weight[0] = Fq*params['Qi']*1e6
                    inh_netcon.append(netcon)

                    # then just the area to check
                    area_tot+=nrn.area(seg.x, sec=section)
                    area_list.append(nrn.area(seg.x, sec=section))
                        

        exc_synapses.append(exc_synapse)
        exc_netcons.append(exc_netcon)
        exc_Ks.append(exc_K)
        exc_spike_trains.append(exc_spike_train)
        inh_synapses.append(inh_synapse)
        inh_netcons.append(inh_netcon)
        inh_Ks.append(inh_K)
        inh_spike_trains.append(inh_spike_train)
        area_lists.append(area_list)

        
    # ## --- spiking properties
    if spiking_mech: # only at soma if we want it !
        nrn('cable_0_0 insert hh')
        # + spikes recording !
        counter = nrn.APCount(0.5, sec=nrn.cable_0_0)
        spkout = nrn.Vector()
        counter.thresh = -30
        counter.record(spkout)
    else:
        spkout = np.zeros(0)
        
    print("=======================================")
    print(" --- checking if the neuron is created ")
    nrn.topology()
    print(("with the total surface : ", area_tot))
    print("=======================================")
        
    return exc_synapses, exc_netcons, exc_Ks, exc_spike_trains,\
       inh_synapses, inh_netcons, inh_Ks, inh_spike_trains, area_lists, spkout

def get_v(cables):
    """
    get all the membrane potential values at on emoment in time
    """
    v = []
    
    v.append(np.array([[nrn.cable_0_0(.5)._ref_v[0]]])) # somatic potential
    
    # now dendritic potentials
    for level in range(1, len(cables)):
        v0 = []
        for comp in range(max(1,2**(level-1))):
            v1 = []
            exec('section = nrn.cable_'+str(level)+'_'+str(comp))
            for seg in section:
                exec('v1.append(nrn.cable_'+str(level)+'_'+str(comp)+'('+\
                     str(seg.x)+')._ref_v[0])')
            v0.append(np.array(v1))
        v.append(np.array(v0))
        
    return v

def set_presynaptic_spikes_manually(shotnoise_input, cables, params,\
                                    exc_spike_trains, exc_Ks,
                                    inh_spike_trains, inh_Ks, tstop, seed=2,\
                                    synchronous_stim={'location':[4,14], 'spikes':[]}):

    synchrony = shotnoise_input['synchrony']
    
    for i in range(len(cables)):
        for j in range(len(exc_spike_trains[i])):
            jj = j%(cables[i]['NSEG'])

            tree_fraction = float(jj)/(len(cables)-1)/cables[i]['NSEG']+float(i-1)/(len(cables)-1)
            if tree_fraction>params['fraction_for_L_prox']:
                fe, fi = shotnoise_input['fe_dist'], shotnoise_input['fi_dist']
            else:
                fe, fi = shotnoise_input['fe_prox'], shotnoise_input['fi_prox']
            
            ## excitation
            build_poisson_spike_train(exc_spike_trains[i][j], fe, exc_Ks[i][j], units='ms',\
                                      tstop=tstop, seed=i*(seed**2+j), synchrony=synchrony)
            if [i,j]==synchronous_stim['location']:
                for tspk in synchronous_stim['spikes']:
                    exc_spike_trains[i][j].append(tspk)
                exc_spike_trains[i][j] = np.sort(exc_spike_trains[i][j])
                
            ## inhibition
            build_poisson_spike_train(inh_spike_trains[i][j], fi, inh_Ks[i][j], units='ms',\
                                      tstop=tstop, seed=seed+i*j**2, synchrony=synchrony)

def run_simulation(shotnoise_input, cables, params, tstop=2000.,\
                   dt=0.025, seed=3, recordings='full', recordings2='', nmda_on=False, Ca_spikes_on=False, HH_on=False,\
                   synchronous_stim={'location':[4,14], 'spikes':[]}):
    """
    recordings is a set of tuple of the form : [branch_generation, branch_number, xseg]
    """
    exc_synapses, exc_netcons, exc_Ks, exc_spike_trains,\
       inh_synapses, inh_netcons, inh_Ks, inh_spike_trains,\
       area_lists, spkout = Constructing_the_ball_and_tree(params, cables, nmda_on=nmda_on, Ca_spikes_on=Ca_spikes_on, HH_on=HH_on)

    # then synapses manually
    set_presynaptic_spikes_manually(shotnoise_input, cables, params,\
                                    exc_spike_trains, exc_Ks,
                                    inh_spike_trains, inh_Ks, tstop, seed=seed,\
                                    synchronous_stim=synchronous_stim)
    
    ## QUEUING OF PRESYNAPTIC EVENTS
    init_spike_train = queue_presynaptic_events_in_NEURON([exc_netcons, exc_spike_trains, inh_netcons, inh_spike_trains])
    ## --- launching the simulation
    fih = nrn.FInitializeHandler((init_spike_train, [exc_netcons, exc_spike_trains, inh_netcons, inh_spike_trains]))
    
    ## --- recording
    t_vec = nrn.Vector()
    t_vec.record(nrn._ref_t)
    V = []

    if recordings is 'soma':
        V.append(nrn.Vector())
        exec('V[0].record(nrn.cable_0_0(0)._ref_v)')
    if recordings2 is 'cable_end':
        V.append(nrn.Vector())
        exec('V[1].record(nrn.cable_5_15(1)._ref_v)')

    ## --- launching the simulation
    nrn.finitialize(params['El']*1e3)
    nrn.dt = dt

    if recordings is 'full':
        V.append(get_v(cables))

    while nrn.t<(tstop-dt):
        nrn.fadvance()
        if recordings is 'full':
            V.append(get_v(cables))

    print("=======================================")
    nrn('forall delete_section()')
    print(" --- checking if the neuron is destroyed")
    nrn.topology()
    print("=======================================")
    return np.array(t_vec), V

