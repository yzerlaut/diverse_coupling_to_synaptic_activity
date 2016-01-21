import numpy as np

def build_poisson_spike_train(spk_train, f, K, units='ms', tstop=2000., seed=1):
    np.random.seed(seed)
    f+=1e-12;K+=1e-12
    if units=='ms':
        factor=1e3
    else:
        factor=1
    spk_train.append(np.random.exponential(factor/f/K, 1)[0])
    while spk_train[-1]<tstop:
        spk_train.append(spk_train[-1]+np.random.exponential(factor/f/K, 1)[0])
    if spk_train[-1]>tstop:
        spk_train[-1]=tstop

def add_synchronous_spikes(synchrony, spk_trains, Ks, Ktot, seed=1):
    np.random.seed(seed)
    Ntot_spikes, Ntot_coincid = 0, 0
    for spk_trains1, Ks1 in zip(spk_trains, Ks):
        for spk_train1, K1 in zip(spk_trains1, Ks1):
            rdm = np.random.uniform(0, 1, size=(len(spk_train1), len(Ks)))
            for spikes, i1 in zip(spk_train1, range(len(spk_train1))):
                Ntot_spikes +=1
                for spk_trains2, Ks2 in zip(spk_trains, Ks):
                    for spk_train2, K2, i2 in zip(spk_trains2, Ks2, range(len(Ks2))):
                        print K1, K2, Ktot
                        if rdm[i1,i2]<float(synchrony)*K2/K1/Ktot:
                            Ntot_coincid +=1
                            spk_train2.append(spikes)
    print Ntot_spikes, Ntot_coincid
    
                    
def queue_presynaptic_events_in_NEURON(ALL):
    """ we queue here the presynaptic events """
    
    def init_spike_train(ALL):
        exc_netcons, exc_spike_trains, inh_netcons, inh_spike_trains = ALL
        for exc_netcon, exc_spike_train in zip(exc_netcons, exc_spike_trains):
            for netcon, spk_train in zip(exc_netcon, exc_spike_train):
                for t in spk_train: # loop over events
                    netcon.event(t)
        for inh_netcon, inh_spike_train in zip(inh_netcons, inh_spike_trains):
            for netcon, spk_train in zip(inh_netcon, inh_spike_train):
                for t in spk_train: # loop over events
                    netcon.event(t)
        return True
    return init_spike_train









