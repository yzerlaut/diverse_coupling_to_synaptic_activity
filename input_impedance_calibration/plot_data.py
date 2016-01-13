import numpy as np
import matplotlib.pylab as plt
import os, sys
sys.path.append('/home/yann/work/python_library/')
import my_graph as graph


def make_experimental_fig():
    fig, AX = plt.subplots(1, 2, figsize=(11,4))
    plt.subplots_adjust(bottom=.25)
    MICE, RATS = [], []

    psd_boundaries = [200,900]
    all_freq, all_psd, all_phase = [np.empty(0, dtype=float) for i in range(3)]
    HIGH_BOUND = 450 # higher nound for frequency

    for file in os.listdir("./intracellular_data/"):
        if file.endswith("_rat.txt"):
            _ = 0
        elif file.endswith(".txt"):
            freq, psd, phase = np.loadtxt("./intracellular_data/"+file, unpack=True)
            MICE.append({'freq':freq[np.argsort(freq)], 'psd':psd[np.argsort(freq)], 'phase':(phase[np.argsort(freq)]%(2.*np.pi)+np.pi/2.)%(2.*np.pi)-np.pi/2.})
            # print file, MICE[-1]['phase'].max()
            all_freq = np.concatenate([all_freq, MICE[-1]['freq']])
            all_psd = np.concatenate([all_psd, MICE[-1]['psd']])
            all_phase = np.concatenate([all_phase, MICE[-1]['phase']])

        psd_boundaries[0] = min([psd_boundaries[0], psd.max()])
        psd_boundaries[1] = max([psd_boundaries[1], psd.max()])

    np.save('full_data.npy',\
        [all_freq[all_freq<HIGH_BOUND], all_psd[all_freq<HIGH_BOUND], all_phase[all_freq<HIGH_BOUND]])
    all_freq, all_psd, all_phase = np.load('full_data.npy')

    mymap = graph.get_linear_colormap()
    X = 300+np.arange(3)*300

    for m in MICE:
        rm = m['psd'][:3].mean()
        r = (rm-psd_boundaries[0])/(psd_boundaries[1]-psd_boundaries[0])
        AX[0].loglog(m['freq'][m['freq']<HIGH_BOUND], m['psd'][m['freq']<HIGH_BOUND], 'D-', color=mymap(r,1), ms=4)
        AX[1].semilogx(m['freq'][m['freq']<HIGH_BOUND], m['phase'][m['freq']<HIGH_BOUND], 'D-', color=mymap(r,1), ms=4)

    AX[0].annotate('n='+str(len(MICE))+' cells', (.2,.2), xycoords='axes fraction', fontsize=18)

    ax2 = fig.add_axes([0.59,0.7,0.015,0.18])
    cb = graph.build_bar_legend(X, ax2, mymap, scale='linear', color_discretization=100,\
                                bounds=psd_boundaries,
                                label=r'$R_\mathrm{m}$ ($\mathrm{M}\Omega$)')



    graph.set_plot(AX[0], xlim=[.08,1000], ylim=[6., 1200],\
               xticks=[1,10,100,1000], yticks=[10,100,1000],yticks_labels=['10','100','1000'],\
                   xlabel='frequency (Hz)',ylabel='input impedance ($\mathrm{M}\Omega$)')

    graph.set_plot(AX[1], xlim=[.08,1000], ylim=[-.2,2.3],\
                   xticks=[1,10,100,1000], yticks=[0,np.pi/4.,np.pi/2.],\
                   yticks_labels=[0,'$\pi/4$', '$\pi/2$'],
                   xlabel='frequency (Hz)', ylabel='phase shift (Rd)')


if __name__=='__main__':

    fig = make_experimental_fig()
    plt.show()
