import numpy as np

def build_poisson_spike_train(spk_train, f, K, units='ms', tstop=2000., seed=1, synchrony=0):
    f_correct = f/(1.+synchrony)
    np.random.seed(seed)
    f_correct+=1e-12;K+=1e-12
    if units=='ms':
        factor=1e3
    else:
        factor=1
    spk_train.append(np.random.exponential(factor/f_correct/K, 1)[0])
    while spk_train[-1]<tstop:
        spk_train.append(spk_train[-1]+np.random.exponential(factor/f_correct/K, 1)[0])
        if np.random.rand()<synchrony: # then we double the spike
            print 'synchronous events'
            spk_train.append(spk_train[-1])
    spk_train[-1] = tstop


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









