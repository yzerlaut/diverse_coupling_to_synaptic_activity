import numpy as np
import matplotlib.pylab as plt
import os, sys
sys.path.append('../code')
import my_graph as graph
sys.path.append('../')
from theory.analytical_calculus import * # where the core calculus lies
from .get_calib import *
from data_firing_response.analyze_data import get_Rm_range

Rm_exp = get_Rm_range()[kept_cells]

imin = np.argmin(Rm_exp)
imax = np.argmax(Rm_exp)
print(imin, imax)
FIGS = []
AX = []

from theory.brt_drawing import make_fig # where the core calculus lies
for i in [imin, imax]:
    soma1, stick1, params1 = adjust_model_prop(Rm_exp[i], soma, stick,\
                                               precision=.01, maxiter=1e5)
    fig, ax = make_fig(np.linspace(0, 1, stick1['B']+1)*stick1['L'],
                        stick1['D'], xscale=1e-6, yscale=50e-6)
    FIGS.append(fig)
            
graph.put_list_of_figs_to_svg_fig(FIGS, visualize=False)
