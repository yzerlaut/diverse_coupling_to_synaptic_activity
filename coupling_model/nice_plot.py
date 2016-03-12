import numpy as np
import sys
sys.path.append('../')
from scipy.stats.stats import pearsonr
import matplotlib.pylab as plt
sys.path.append('../code')
from my_graph import set_plot, put_list_of_figs_to_svg_fig

import matplotlib
font = {'size':24}
matplotlib.rc('font', **font)

COUPLINGS, BIOPHYSICS = np.load('data/elctrophy_vs_coupling.npy')
E_LABELS = [r"$\langle V_\mathrm{thre}^\mathrm{eff} \rangle_\mathcal{D}$ (mV)",\
            r"$\langle \partial \nu / \partial \mu_V \rangle_\mathcal{D}$ (Hz/mV)",\
            r"$\langle \partial \nu / \partial \sigma_V \rangle_\mathcal{D}$ (Hz/mV)",\
            r"$\langle \partial \nu / \partial \tau_V^{N} \rangle_\mathcal{D}$ (Hz/%)"]

VTHRE, DMUV, DTSV, DTV = [BIOPHYSICS[i,:] for i in range(BIOPHYSICS.shape[0])]
NU0, NU, UNBALANCED, PROX, DIST, SYNCH = [COUPLINGS[i,:] for i in range(COUPLINGS.shape[0])]

# special plot of the highlighted cells !!
INDEXES, MARKER, SIZE = [0, 2, 27, 1], ['^', 'd', '*', 's'], [12, 11, 17, 10]
NU0s, NUs, UNBALANCEDs, PROXs, DISTs, SYNCHs = NU0[INDEXES], NU[INDEXES], UNBALANCED[INDEXES], PROX[INDEXES], DIST[INDEXES], SYNCH[INDEXES]
VTHREs, DMUVs, DTSVs, DTVs = VTHRE[INDEXES], DMUV[INDEXES], DTSV[INDEXES], DTV[INDEXES]

## discarding too low firing that can't be analyzed...

# cond = (NU0>1e-3) & (NU0<8)
cond = (NU0>0)# to have all cells !!
cond = (NU0>1e-4)
np.save('kept_cells.npy', cond)
print 'number of kept cells:',  len(NU0[cond])
NU0, NU, UNBALANCED, PROX, DIST, SYNCH = NU0[cond], NU[cond], UNBALANCED[cond], PROX[cond], DIST[cond], SYNCH[cond]
VTHRE, DMUV, DTSV, DTV = VTHRE[cond], DMUV[cond], DTSV[cond], DTV[cond]


def plot_all(ax, X, Y, lin_fit, Xs, Ys, cc, pp, invert_axis=False):
    for k in range(len(Xs)):
        ax.plot([Xs[k]], [Ys[k]], color='lightgray', marker=MARKER[k], label='cell '+str(k), ms=SIZE[k])
    x = np.linspace(X.min(), X.max())
    y = np.polyval(lin_fit, x)
    ax.plot(x, y, 'k--', lw=.5)
    ax.plot(X, Y, 'ko')
    ax.annotate('c='+str(np.round(cc,2))+', '+'p='+'%.1e' % pp,\
                         (0.05,1.1), xycoords='axes fraction', fontsize=24)
    if invert_axis:
        ax.invert_xaxis()


fig, AX = plt.subplots(6, 5, figsize=(22,30))
fig.subplots_adjust(wspace=.6, hspace=1.)

######################################################################
######## baseline activity
######################################################################

## mean response -- HISTOGRAM
y = np.log(NU)/np.log(10)
ys = np.log(NUs)/np.log(10)
yy = np.log(COUPLINGS[0,:][(1e-2<COUPLINGS[0,:]) & (COUPLINGS[0,:]<1e3)])/np.log(10)
AX[0,0].hist(yy, bins=np.linspace(-2.5,1.05,9), color='lightgray', edgecolor='k', lw=2)
AX[0,0].plot([1.5],[0], 'wD', alpha=0, ms=0.01)
set_plot(AX[0,0], ['left', 'bottom'], ylabel='cell #',\
    xlabel='baseline activity \n'+r'$ \nu_\mathrm{bsl}$ (Hz)',\
    xticks=[-1,0,1], yticks=[0,3,6], xlim=[-2.7,1.5],\
    xticks_labels=['0.1', '1 ', '10'])
## -- CORRELATION WITH VTHRE
cc, pp = pearsonr(VTHRE, y)
lin_fit = np.polyfit(np.array(VTHRE, dtype='f8'), np.array(y, dtype='f8'), 1)
plot_all(AX[0,1], VTHRE, y, lin_fit, VTHREs, ys,\
         cc, pp, invert_axis=True)
set_plot(AX[0,1], ['left', 'bottom'],\
    ylabel=r'$ \nu_\mathrm{bsl}$ (Hz)',yticks=[-1,0,1],\
    yticks_labels=['0.1', '1 ', '10'], xticks=[-40, -50, -60],\
    xticks_labels=[])
## -- CORRELATION WITH THE REST
for X, Xs, ax, label in zip([DMUV, DTSV, DTV], [DMUVs, DTSVs, DTVs], AX[0,2:], E_LABELS[1:]):
    cc, pp = pearsonr(X, y)
    lin_fit = np.polyfit(np.array(X, dtype='f8'),np.array(y, dtype='f8'), 1)
    plot_all(ax, X, y, lin_fit, Xs,\
             ys, cc, pp, invert_axis=(label==E_LABELS[-1]))
    set_plot(ax, ['left', 'bottom'],\
             yticks=[-1,0,1],\
             xticks_labels=[], yticks_labels=[], num_xticks=3)

######################################################################
######## unbalanced activity
######################################################################

## unbalanced activity -- HISTOGRAM
y = np.log(UNBALANCED)/np.log(10)
ys = np.log(UNBALANCEDs)/np.log(10)

AX[1,0].hist(y, bins=7, color='lightgray', edgecolor='k', lw=2)
AX[1,0].plot([1.5],[0], 'wD', alpha=0, ms=0.01)
set_plot(AX[1,0], ['left', 'bottom'], ylabel='cell #',\
    xlabel='response to unbalanced \n'+r' activity, $\delta \nu_\mathrm{ubl}$ (Hz)',\
    xticks=[-1,0,1], yticks=[0,3,6], xlim=[-1.9,1.8],\
    xticks_labels=['0.1', '1 ', '10'])
## unbalanced activity -- CORRELATION WITH VTHRE
cc, pp = pearsonr(VTHRE, y)
lin_fit = np.polyfit(np.array(VTHRE, dtype='f8'),\
                     np.array(y, dtype='f8'), 1)
plot_all(AX[1,1], VTHRE, y, lin_fit, VTHREs,\
         ys, cc, pp, invert_axis=True)
set_plot(AX[1,1], ['left', 'bottom'],ylabel=r'$\delta \nu_\mathrm{ubl}$ (Hz)',\
    yticks=[-1,0,1], yticks_labels=['0.1', '1 ', '10'], xticks=[-40, -50, -60], xticks_labels=[])
## unbalanced activity -- CORRELATION WITH THE REST
for X, Xs, ax, label in zip([DMUV, DTSV, DTV], [DMUVs, DTSVs, DTVs], AX[1,2:], E_LABELS[1:]):
    cc, pp = pearsonr(X, y)
    lin_fit = np.polyfit(np.array(X, dtype='f8'),\
                         np.array(y, dtype='f8'), 1)
    plot_all(ax, X, y, lin_fit, Xs,\
             ys, cc, pp, invert_axis=(label==E_LABELS[-1]))
    set_plot(ax, ['left', 'bottom'],\
             xticks_labels=[], yticks=[-1,0,1],\
             yticks_labels=[], num_xticks=3)

######################################################################
######## proximal activity
######################################################################

y = (PROX-NU0) #np.log(PROX+NU)/np.log(10)
ys = (PROXs-NU0s) #np.log(PROXs+NUs)/np.log(10)

AX[2,0].hist(y, bins=10, color='lightgray', edgecolor='k', lw=2)
set_plot(AX[2,0], ['left', 'bottom'], ylabel='cell #',\
    xlabel='response to proximal \n'+r' activity, $\delta \nu_\mathrm{prox}$ (Hz)')
##  -- CORRELATION WITH VTHRE

y = 100.*(PROX-NU0)/NU #np.log(PROX+NU)/np.log(10)
ys = 100.*(PROXs-NU0s)/NUs #np.log(PROXs+NUs)/np.log(10)

cc, pp = pearsonr(VTHRE, y)
lin_fit = np.polyfit(np.array(VTHRE, dtype='f8'), np.array(y, dtype='f8'), 1)
plot_all(AX[2,1], VTHRE, y, lin_fit, VTHREs, ys, cc, pp, invert_axis=True)
set_plot(AX[2,1], ['left', 'bottom'], num_yticks=3,\
         ylabel=r'$\delta \nu_\mathrm{prox}$/$\nu_\mathrm{bsl}$ (%)', xticks=[-40, -50, -60],\
         xticks_labels=[])
##  -- CORRELATION WITH THE REST
for X, Xs, ax, label in zip([DMUV, DTSV, DTV], [DMUVs, DTSVs, DTVs], AX[2,2:], E_LABELS[1:]):
    cc, pp = pearsonr(X, y)
    lin_fit = np.polyfit(np.array(X, dtype='f8'),\
                         np.array(y, dtype='f8'), 1)
    plot_all(ax, X, y, lin_fit, Xs,\
             ys, cc, pp, invert_axis=(label==E_LABELS[-1]))
    set_plot(ax, ['left', 'bottom'],\
             num_yticks=3, num_xticks=3, yticks_labels=[], xticks_labels=[])
             

######################################################################
######## distal activity
######################################################################

y = np.log(DIST)/np.log(10)
ys = np.log(DISTs)/np.log(10)

AX[3,0].hist(y, bins=9, color='lightgray', edgecolor='k', lw=2)
set_plot(AX[3,0], ['left', 'bottom'], ylabel='cell #',\
         xticks=[-1,0,1], xticks_labels=['0.1', '1 ', '10'],
         xlabel='response to distal \n'+r' activity, $\delta \nu_\mathrm{dist}$ (Hz)')
## unbalanced activity -- CORRELATION WITH VTHRE
cc, pp = pearsonr(VTHRE, y)
lin_fit = np.polyfit(np.array(VTHRE, dtype='f8'), np.array(y, dtype='f8'), 1)
plot_all(AX[3,1], VTHRE, y, lin_fit, VTHREs, ys, cc, pp, invert_axis=True)
set_plot(AX[3,1], ['left', 'bottom'],\
         yticks=[-1,0,1], yticks_labels=['0.1', '1 ', '10'],
         xticks=[-40, -50, -60], xticks_labels=[],\
         ylabel=r'$\delta \nu_\mathrm{dist}$ (Hz)')
## unbalanced activity -- CORRELATION WITH THE REST
for X, Xs, ax, label in zip([DMUV, DTSV, DTV], [DMUVs, DTSVs, DTVs], AX[3,2:], E_LABELS[1:]):
    cc, pp = pearsonr(X, y)
    lin_fit = np.polyfit(np.array(X, dtype='f8'),\
                         np.array(y, dtype='f8'), 1)
    plot_all(ax, X, y, lin_fit, Xs,\
             ys, cc, pp, invert_axis=(label==E_LABELS[-1]))
    set_plot(ax, ['left', 'bottom'],\
             yticks=[-1,0,1], yticks_labels=[],
             xticks_labels=[], num_xticks=3)
             
######################################################################
######## synchronyzed activity
######################################################################

y = np.log(SYNCH)/np.log(10)
ys = np.log(SYNCHs)/np.log(10)

AX[4,0].hist(y, bins=9, color='lightgray', edgecolor='k', lw=2)
set_plot(AX[4,0], ['left', 'bottom'], ylabel='cell #',\
         xticks=[-1,0,1], xticks_labels=['0.1', '1 ', '10'],
         xlabel='response to synchrony \n'+r'$\delta \nu_\mathrm{synch}$ (Hz)')
## -- CORRELATION WITH VTHRE
cc, pp = pearsonr(VTHRE, y)
lin_fit = np.polyfit(np.array(VTHRE, dtype='f8'), np.array(y, dtype='f8'), 1)
plot_all(AX[4,1], VTHRE, y, lin_fit, VTHREs, ys, cc, pp, invert_axis=True)
set_plot(AX[4,1], ['left', 'bottom'], xlabel=E_LABELS[0],\
         yticks=[-1,0,1], yticks_labels=['0.1', '1 ', '10'],
         xticks=[-40, -50, -60], ylabel=r'$\delta \nu_\mathrm{synch}$ (Hz)')
##  -- CORRELATION WITH THE REST
for X, Xs, ax, label in zip([DMUV, DTSV, DTV], [DMUVs, DTSVs, DTVs], AX[4,2:], E_LABELS[1:]):
    cc, pp = pearsonr(X, y)
    lin_fit = np.polyfit(np.array(X, dtype='f8'),\
                         np.array(y, dtype='f8'), 1)
    plot_all(ax, X, y, lin_fit, Xs,\
             ys, cc, pp, invert_axis=(label==E_LABELS[-1]))
    set_plot(ax, ['left', 'bottom'], xlabel=label,\
             yticks=[-1,0,1], yticks_labels=[],
             num_xticks=3)

plt.annotate('increasing \n excitability', (0.1,0.1), xycoords='figure fraction')
plt.annotate('increasing \n sensitivity to $\mu_V$', (0.1,0.1), xycoords='figure fraction')
plt.annotate('increasing \n sensitivity to $\sigma_V$', (0.1,0.1), xycoords='figure fraction')
plt.annotate('increasing \n sensitivity to $\tau_V$', (0.1,0.1), xycoords='figure fraction')

# plt.show()
fig.savefig('final_fig.svg')
fig.savefig('final_fig.png')

