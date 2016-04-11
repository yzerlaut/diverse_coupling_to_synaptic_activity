import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

def vec_diameter(D,EqCylinder):
    """ returns the different diameters of the equivalent cylinder
    given a number of branches point"""
    DVEC = []
    DVEC.append(D)
    for i in range(1,len(EqCylinder)-1):
        D0 = D*2**(-2.*i/3.)
        DVEC.append(D0)
    return DVEC


def make_fig(EqCylinder, diam_stick,\
             xscale=2e-6, yscale=200e-6,\
             added_points=None, color='k'):

    fig, ax = plt.subplots(1, figsize=(.1*len(EqCylinder)**2,6), frameon=False)
    ax.axis('off')

    vecD = vec_diameter(diam_stick, EqCylinder)

    # let's start the figure
    patches, links = [], []
    D0 = vecD[-1]# scale as 1
    DY = EqCylinder[-1]/10./len(EqCylinder)
    nmax = 2**(len(EqCylinder)-2)
    # to have the global view, we need to add some points
    ax.plot([-nmax,nmax], [-4*DY,EqCylinder[-1]], 'w', alpha=0.)
    # we start with the last branches
    vec_middle = np.arange(-nmax+1, nmax, 2)
    COORDS = [] # to remember for the points
    for i in range(1,len(EqCylinder)-1)[::-1]:
        COORDS.append(vec_middle)
        DX = 1.*vecD[i]/D0
        L = EqCylinder[i+1]-EqCylinder[i]
        for x in vec_middle:
            patches.append(Rectangle((x-DX/2.,EqCylinder[i]+i*DY), DX, L))
        for x1, x2 in zip(vec_middle[::2], vec_middle[1::2]):
            links.append(Line2D((x1,x2),(EqCylinder[i]+i*DY,\
                                         EqCylinder[i]+i*DY)))
        vec_middle = (vec_middle[:-1]+.5*np.diff(vec_middle))[::2]
        for x in vec_middle:
            links.append(Line2D((x,x),(EqCylinder[i]+(i-1)*DY,\
                                         EqCylinder[i]+i*DY)))

    # then adding the first 
    DX = 1.*vecD[0]/D0
    L = EqCylinder[1]-EqCylinder[0]
    patches.append(Rectangle((-.5*DX, 0), DX, L))
    COORDS.append(np.zeros(1))
    # then adding the soma
    plt.plot([0],[-3*DY],'o', ms=10, color=color)
    links.append(Line2D((0,0),(-3*DY,0)))
    # then x and y scales
    patches.append(Rectangle((nmax-3*xscale/D0, -2*DY), xscale/D0, DY))
    plt.annotate(str(int(xscale*1e6))+'$\mu$m', (nmax-3*xscale/D0, -5*DY))
    patches.append(Rectangle((nmax-2*xscale/D0, -2*DY), .5, yscale))
    plt.annotate(str(int(yscale*1e6))+'$\mu$m', (nmax-xscale/D0, .8*yscale))

    ax.add_collection(PatchCollection(patches, facecolor=color, edgecolor=color))
    ax.add_collection(PatchCollection(links, linewidth=1))

    # we will need to add additional points
    if added_points is not None:
        for points in added_points:
            # points are of the form
            branch_level, branch_number,\
                frac_of_branch, color, marker, ms = points
            X = COORDS[::-1][branch_level-1][branch_number-1]
            Y = (branch_level-1)*DY+EqCylinder[branch_level-1] + \
             frac_of_branch*(EqCylinder[branch_level]-EqCylinder[branch_level-1])
            plt.plot([X], [Y], color+marker, ms=ms)
    return fig, ax

if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser(description=
     """ 
     description of the whole modulus here
     """
    ,formatter_class=argparse.RawTextHelpFormatter)

    # ball and tree properties
    parser.add_argument("--L_stick", type=float, help="Length of the stick in micrometer", default=700.)
    parser.add_argument("--D_stick", type=float, help="Diameter of the stick", default=2.)
    parser.add_argument("-B", "--branches", type=int, help="Number of branches (equally spaced)", default=5)
    parser.add_argument("--L_proximal", type=float, help="Length of the proximal compartment", default=2000.)

    args = parser.parse_args()

    EqCylinder = np.linspace(0,args.L_stick,args.branches+1)*1e-6
    # EqCylinder = 1e-6*np.array([0, 100, 500, 600, 1000, 1800, 2000])
    # EqCylinder = 1e-6*np.array([0, 100, 600, 2000])
    # EqCylinder = 1e-6*np.array([0, 2000])

    # fig, ax = make_fig(EqCylinder, args.D_stick,\
    #                    added_points=[[5,8,.1,'g','D', 8],\
    #                                  [6,5,.5,'r','x', 10]])
    fig, ax = make_fig(EqCylinder, args.D_stick)

    plt.show()


