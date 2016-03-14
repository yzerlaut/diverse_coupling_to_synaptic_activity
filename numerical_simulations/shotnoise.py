import numpy as np

def build_poisson_spike_train(spk_train, f, K, units='ms', tstop=2000., seed=1, synchrony=0.):

    np.random.seed(seed)
    if units=='ms':
        factor=1e3
    else:
        factor=1
    if f>1e-9 and K>1e-9:
        new_f = f/(1.+synchrony+synchrony**2+synchrony**3)
        N = int(new_f*K*tstop/factor)+1
        exp_rdm = np.random.exponential(factor/new_f/K, N)
        for event in np.cumsum(exp_rdm): spk_train.append(event)
        uniform_rdm = np.random.random(N)
        for i in np.arange(N)[uniform_rdm<synchrony**3]:
            for j in range(4):
                spk_train.insert(i, spk_train[i]) # 4 events
        # print '4 events:', len(np.arange(N)[uniform_rdm<synchrony**3])
        for i in np.arange(N)[(uniform_rdm>synchrony**3) & (uniform_rdm<synchrony**2)]:
            for j in range(3):
                spk_train.insert(i, spk_train[i]) # 3 events
        # print '3 events:', len(np.arange(N)[(uniform_rdm>synchrony**3) & (uniform_rdm<synchrony**2)])
        for i in np.arange(N)[(uniform_rdm>synchrony**2) & (uniform_rdm<synchrony)]:
            for j in range(2):
                spk_train.insert(i, spk_train[i]) # 2 events
        # print '2 events:', len(np.arange(N)[(uniform_rdm>synchrony**2) & (uniform_rdm<synchrony)])

def queue_presynaptic_events_in_NEURON(ALL):
    """ we queue here the presynaptic events """
    
    def init_spike_train(ALL):
        ie, ii = 0, 0
        exc_netcons, exc_spike_trains, inh_netcons, inh_spike_trains = ALL
        for exc_netcon, exc_spike_train in zip(exc_netcons, exc_spike_trains):
            for netcon, spk_train in zip(exc_netcon, exc_spike_train):
                for t in spk_train: # loop over events
                    netcon.event(t)
                    ie+=1
        for inh_netcon, inh_spike_train in zip(inh_netcons, inh_spike_trains):
            for netcon, spk_train in zip(inh_netcon, inh_spike_train):
                for t in spk_train: # loop over events
                    netcon.event(t)
                    ii+=1
        print ie, ii
        return True
    return init_spike_train









