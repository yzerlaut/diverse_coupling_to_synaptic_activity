
# nrn_simulations.py
from neuron import h as nrn
import numpy as np

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
    print "Total number of EXCITATORY synapses : ", Ke_tot
    print "Total number of INHIBITORY synapses : ", Ki_tot
    return xtot, cables

def Constructing_the_ball_and_tree(params, cables,
                                spiking_mech=False):

    # list of excitatory stuff
    exc_synapses, exc_netcons, exc_netstims = [], [], [] 
    # list of inhibitory stuff
    inh_synapses, inh_netcons, inh_netstims = [], [], []
    # area list
    area_lists = []
    
    ## --- CREATING THE NEURON
    area_tot = 0

    for level in range(len(cables)):
        # list of excitatory stuff per level
        exc_synapse, exc_netcon, exc_netstim = [], [], [] 
        # list of inhibitory stuff per level
        inh_synapse, inh_netcon, inh_netstim = [], [], []
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

                if level>=1: # if not soma
                    # in NEURON : connect daughter, mother branch
                    nrn('connect cable_'+str(int(level))+'_'+\
                    str(int(comp))+'(0), '+\
                    'cable_'+str(int(level-1))+\
                    '_'+str(int(comp/2.))+'(1)')
                    
                ## --- SPREADING THE SYNAPSES (bis)
                for seg in section:

                    # in each segment, we insert an excitatory synapse
                    netstim = nrn.NetStim(seg.x, sec=section)
                    netstim.noise, netstim.number, netstim.start = \
                              1, 1e20, 0

                    exc_netstim.append(netstim)
                    syn = nrn.ExpSyn(seg.x, sec=section)
                    syn.tau, syn.e = params['Te']*1e3, params['Ee']*1e3
                    exc_synapse.append(syn)
                    netcon = nrn.NetCon(netstim, syn)
                    netcon.weight[0] = params['Qe']*1e6
                    exc_netcon.append(netcon)

                    # then in each segment, we insert an inhibitory synapse
                    netstim = nrn.NetStim(seg.x, sec=section)
                    netstim.noise, netstim.number, netstim.start = \
                          1, 1e20, 0

                    inh_netstim.append(netstim)
                    syn = nrn.ExpSyn(seg.x, sec=section)
                    syn.tau, syn.e = params['Ti']*1e3, params['Ei']*1e3
                    inh_synapse.append(syn)
                    netcon = nrn.NetCon(netstim, syn)
                    netcon.weight[0] = params['Qi']*1e6
                    inh_netcon.append(netcon)

                    # then just the area to check
                    area_tot+=nrn.area(seg.x, sec=section)
                    area_list.append(nrn.area(seg.x, sec=section))


        exc_synapses.append(exc_synapse)
        exc_netcons.append(exc_netcon)
        exc_netstims.append(exc_netstim)
        inh_synapses.append(inh_synapse)
        inh_netcons.append(inh_netcon)
        inh_netstims.append(inh_netstim)
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
        
    print "======================================="
    print " --- checking if the neuron is created "
    nrn.topology()
    print "with the total surface : ", area_tot
    print "======================================="
        
    return exc_synapses, exc_netcons, exc_netstims,\
       inh_synapses, inh_netcons, inh_netstims, area_lists, spkout


def run_simulations(fe, fi, cables, params,
                                dt = 0.1, tstop=100.,
                                spiking_mech=True,
                                get_v_of_first_cable=False):
        
    exc_synapses, exc_netcons, exc_netstims,\
       inh_synapses, inh_netcons, inh_netstims,\
       area_lists, spkout =\
          Constructing_the_ball_and_tree(params, cables,
                                spiking_mech=spiking_mech)

    for i in range(len(cables)):
        for j in range(len(exc_netstims[i])):
            # excitation
            Ke = float(area_lists[i][j])/float(cables[i]['exc_density'])
            exc_netstims[i][j].interval = 1e3/fe[i][j]/Ke
            # print exc_interval, exc_netstims[i][j].interval
            # # inhibition
            Ki = area_lists[i][j]/cables[i]['inh_density']
            inh_netstims[i][j].interval = 1e3/fi[i][j]/Ki

            
    ## --- recording
    t_vec = nrn.Vector()
    t_vec.record(nrn._ref_t)
    if get_v_of_first_cable:
        v_vec = np.zeros((int(tstop/dt),cables[1]['NSEG']))
    else:
        v_vec = nrn.Vector()
        v_vec.record(nrn.cable_0_0(0.5)._ref_v)
        
    ## --- launching the simulation
    nrn.finitialize(params['El']*1e3)
    nrn.dt = dt
    if get_v_of_first_cable:
        i=0
        while nrn.t<(tstop-dt):
            j = 0
            for seg in nrn.cable_1_0:
                # v_vec[i,j] = nrn.cable_1_0(seg.x)._ref_v[0]
                j+=1
            i+=1
            nrn.fadvance()
    else:
        while nrn.t<tstop:
            nrn.fadvance()
        
    fout = len(spkout)/nrn.t*1e3

    print "output frequency", fout
    print "======================================="
    nrn('forall delete_section()')
    print " --- checking if the neuron is detroyed"
    nrn.topology()
    print "======================================="
    return np.array(v_vec), np.array(t_vec), fout
