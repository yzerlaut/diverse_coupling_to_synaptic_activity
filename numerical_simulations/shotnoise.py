import numpy as np

def build_poisson_spike_train(spk_train, f, K, units='ms', tstop=2000., seed=1, synchrony=0.):
    np.random.seed(seed)
    if units=='ms':
        factor=1e3
    else:
        factor=1
    if f>1e-9 and K>1e-9:
        new_f, new_K = f/(1.+synchrony), K/(1.+synchrony)
        spk_train.append(np.random.exponential(factor/new_f/new_K, 1)[0])
        while spk_train[-1]<tstop:
            spk_train.append(spk_train[-1]+np.random.exponential(factor/new_f/new_K, 1)[0])

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









