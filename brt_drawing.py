import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection

def vec_diameter(D,EqCylinder):
    """ returns the different diameters of the equivalent cylinder
    given a number of branches point"""
    DVEC = []
    DVEC.append(D)
    for i in range(1,len(EqCylinder)-1):
        D0 = D*2**(-2.*i/3.)
        DVEC.append(D0)
    return DVEC

def make_fig(EqCylinder, vecD):


    fig, ax = plt.subplots(1, figsize=(2.*len(EqCylinder),5), frameon=False)
    ax.axis('off')
    patches = []
    D0 = vecD[-1]# scale as 1
    DY = EqCylinder[-1]/10./len(EqCylinder)
    nmax = 2**(len(EqCylinder)-2)
    ax.plot([-2*(nmax+1),2*(nmax+1)], [-4*DY,EqCylinder[-1]], 'w', alpha=0.)
    # for i in range(len(EqCylinder)-2,len(EqCylinder)-1)[::-1]:
    for i in range(1,len(EqCylinder)-1):
        n = 2**i
        DX = 1.*vecD[i]/D0
        L = EqCylinder[i+1]-EqCylinder[i]
        for j in range(n/2):
            patches.append(Rectangle((-(2*j+1.5)*DX,EqCylinder[i]+i*DY), DX, L))
            patches.append(Rectangle(((2*j+.5)*DX,EqCylinder[i]+i*DY), DX, L))
    # then adding the first 
    DX = 1.*vecD[0]/D0
    L = EqCylinder[1]-EqCylinder[0]
    patches.append(Rectangle((-.5*DX, 0), DX, L))

    # then adding the soma
    plt.plot([0],[0],'ko', ms=10)
    
    ax.add_collection(PatchCollection(patches, facecolor='k'))

    return fig, ax

if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser(description=
     """ 
     description of the whole modulus here
     """
    ,formatter_class=argparse.RawTextHelpFormatter)

    # ball and tree properties
    parser.add_argument("--L_stick", type=float, help="Length of the stick in micrometer", default=2000.)
    parser.add_argument("--D_stick", type=float, help="Diameter of the stick", default=2.)
    parser.add_argument("-B", "--branches", type=int, help="Number of branches (equally spaced)", default=1)
    parser.add_argument("--L_proximal", type=float, help="Length of the proximal compartment", default=2000.)

    EqCylinder = 1e-6*np.array([0, 100, 500, 600, 1000, 1800, 2000])

    # EqCylinder = [0, 1, 2]
    # DX = [0,3, 1, .5]

    vecD = vec_diameter(.2,EqCylinder)

    fig, ax = make_fig(EqCylinder, vecD)

    plt.show()


#  LocalWords:  vecD
