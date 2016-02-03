from numba import jit
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import sys
sys.path.append('/home/yann/work/python_library/')
import fourier_for_real as rfft


def params_for_cable_theory(cable, Params):
    """
    applies the radial symmetry, to get the linear cable theory parameters
    """
    D = cable['D']
    cable['rm'] = 1./Params['g_pas']/(np.pi*D) # [O.m]   """" NEURON 1e-1 !!!! """" 
    cable['ri'] = Params['Ra']/(np.pi*D**2/4) # [O/m]
    cable['cm'] = Params['cm']*np.pi*D # [F/m] """" NEURON 1e-2 !!!! """"
    

def setup_model(EqCylinder, soma, dend, Params):
    """ returns the different diameters of the equivalent cylinder
    given a number of branches point"""
    cables, xtot = [], np.zeros(1)
    cables.append(soma.copy())
    cables[0]['inh_density'] = soma['inh_density']
    cables[0]['exc_density'] = soma['exc_density']
    Ke_tot, Ki_tot = 0, 0
    D = dend['D'] # mothers branch diameter
    for i in range(1,len(EqCylinder)):
        cable = dend.copy()
        cable['x1'], cable['x2'] = EqCylinder[i-1], EqCylinder[i]
        cable['L'] = cable['x2']-cable['x1']
        x = np.linspace(cable['x1'], cable['x2'], cable['NSEG']+1)
        cable['x'] = .5*(x[1:]+x[:-1])
        xtot = np.concatenate([xtot, cable['x']])
        cable['D'] = D*2**(-2*(i-1)/3.)
        cable['inh_density'] = dend['inh_density']
        cable['exc_density'] = dend['exc_density']
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
    # we store this info in the somatic comp
    cables[0]['Ke_tot'], cables[0]['Ki_tot'] = Ke_tot, Ki_tot
    return xtot, cables

def calculate_mean_conductances(shtn_input,\
                                soma, cable, Params):
    """
    this calculates the conductances that needs to plugged in into the
    linear cable equation
    Note :
    the two first conductances are RADIAL conductances !! (excitation and
    inhibition along the cable)
    the third one is an ABSOLUTE conductance !
    """

    D = cable['D']
        
    Te_prox, Qe_prox = Params['Te'], Params['Qe']
    Ti_prox, Qi_prox = Params['Ti'], Params['Qi']
    Qe_dist = Qe_prox*Params['factor_for_distal_synapses_weight']
    Qi_dist = Qi_prox*Params['factor_for_distal_synapses_weight']
    Te_dist = Te_prox*Params['factor_for_distal_synapses_weight']
    Ti_dist = Ti_prox*Params['factor_for_distal_synapses_weight']
        
    ge_prox = Qe_prox*shtn_input['fe_prox']*Te_prox*D*np.pi/cable['exc_density']
    gi_prox = Qi_prox*shtn_input['fi_prox']*Ti_prox*D*np.pi/cable['inh_density']
    ge_dist = Qe_dist*shtn_input['fe_dist']*Te_dist*D*np.pi/cable['exc_density']
    gi_dist = Qi_dist*shtn_input['fi_dist']*Ti_dist*D*np.pi/cable['inh_density']

    # somatic inhibitory conductance
    Ls, Ds = soma['L'], soma['D']
    Kis = np.pi*Ls*Ds/soma['inh_density']
    Gi_soma = Qi_prox*Kis*shtn_input['fi_soma']*Ti_prox
    
    return Gi_soma, ge_prox, gi_prox, ge_dist, gi_dist


def ball_and_stick_params(soma, stick, Params):
    Ls, Ds = soma['L'], soma['D']
    L, D = stick['L'], stick['D']
    Lp = L*Params['fraction_for_L_prox']
    Rm = 1./(np.pi*Ls*Ds*Params['g_pas'])
    # print 'Rm (soma)', 1e-6*Rm, 'MOhm'
    Cm = np.pi*Ls*Ds*Params['cm']
    # print 'Cm (soma)', 1e-12*Cm, 'pF'
    El, Ei, Ee = Params['El'], Params['Ei'], Params['Ee']
    rm, cm, ri = stick['rm'], stick['cm'], stick['ri']
    return Ls, Ds, L, D, Lp, Rm, Cm, El, Ee, Ei, rm, cm, ri


def ball_and_stick_constants(shtn_input, soma, stick, Params):
    Ls, Ds, L, D, Lp, Rm, Cm, El, Ee, Ei, rm, cm, ri = \
                    ball_and_stick_params(soma, stick, Params)
    Gi_soma, ge_prox, gi_prox, ge_dist, gi_dist = \
                calculate_mean_conductances(shtn_input,\
                                            soma, stick, Params)
    tauP = rm*cm/(1+rm*ge_prox+rm*gi_prox)
    lbdP = np.sqrt(rm/ri/(1+rm*ge_prox+rm*gi_prox))
    tauD = rm*cm/(1+rm*ge_dist+rm*gi_dist)
    lbdD = np.sqrt(rm/ri/(1+rm*ge_dist+rm*gi_dist))
    tauS = Rm*Cm/(1+Rm*Gi_soma)
    return tauS, tauP, lbdP, tauD, lbdD


############### INPUT FROM SYMPY ###################
# --- muV
@jit
def A_coeff(Lp, L, lbdD, lbdP, gP, v0D, v0P, V0): return (V0*gP*lbdD*np.cosh(Lp)*np.cosh(L - Lp) + V0*gP*lbdP*np.sinh(Lp)*np.sinh(L - Lp) - gP*lbdD*v0P*np.cosh(Lp)*np.cosh(L - Lp) - gP*lbdP*v0P*np.sinh(Lp)*np.sinh(L - Lp) + lbdP*v0D*np.sinh(L - Lp) - lbdP*v0P*np.sinh(L - Lp))/(gP*lbdD*np.cosh(Lp)*np.cosh(L - Lp) + gP*lbdP*np.sinh(Lp)*np.sinh(L - Lp) + lbdD*np.sinh(Lp)*np.cosh(L - Lp) + lbdP*np.sinh(L - Lp)*np.cosh(Lp))
@jit
def B_coeff(Lp, L, lbdD, lbdP, gP, v0D, v0P, V0): return lbdD*(V0*gP - gP*v0D*np.cosh(Lp) + gP*v0P*np.cosh(Lp) - gP*v0P - v0D*np.sinh(Lp) + v0P*np.sinh(Lp))/(gP*lbdD*np.cosh(Lp)*np.cosh(L - Lp) + gP*lbdP*np.sinh(Lp)*np.sinh(L - Lp) + lbdD*np.sinh(Lp)*np.cosh(L - Lp) + lbdP*np.sinh(L - Lp)*np.cosh(Lp))
@jit
def muV_prox(X, Lp, L, lbdD, lbdP, gP, v0D, v0P, V0): return v0P + A_coeff(Lp, L, lbdD, lbdP, gP, v0D, v0P, V0)*np.cosh(X)+gP*(v0P-V0+A_coeff(Lp, L, lbdD, lbdP, gP, v0D, v0P, V0))*np.sinh(X)
@jit
def muV_dist(X, Lp, L, lbdD, lbdP, gP, v0D, v0P, V0): return v0D + B_coeff(Lp, L, lbdD, lbdP, gP, v0D, v0P, V0)*np.cosh(X-L)

@jit
def Af_coeff(Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP): return If*rfP*(-afD*lbdP*np.cosh(L*afD - Lp*afD - Lp*afP + Xsrc*afP) + afD*lbdP*np.cosh(L*afD - Lp*afD + Lp*afP - Xsrc*afP) + afP*lbdD*np.cosh(L*afD - Lp*afP - Xsrc*afD + Xsrc*afP) + afP*lbdD*np.cosh(L*afD + Lp*afP - Xsrc*afD - Xsrc*afP))/(afP*(-afD*gfP*lbdP*np.cosh(-L*afD + Lp*afD + Lp*afP) + afD*gfP*lbdP*np.cosh(L*afD - Lp*afD + Lp*afP) - afD*lbdP*np.sinh(-L*afD + Lp*afD + Lp*afP) + afD*lbdP*np.sinh(L*afD - Lp*afD + Lp*afP) + afP*gfP*lbdD*np.cosh(-L*afD + Lp*afP + Xsrc*afD) + afP*gfP*lbdD*np.cosh(L*afD + Lp*afP - Xsrc*afD) + afP*lbdD*np.sinh(-L*afD + Lp*afP + Xsrc*afD) + afP*lbdD*np.sinh(L*afD + Lp*afP - Xsrc*afD)))
@jit
def Bf_coeff(Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP): return If*lbdD*rfP*(gfP*np.sinh(-L*afD + Xsrc*afD + Xsrc*afP) + gfP*np.sinh(L*afD - Xsrc*afD + Xsrc*afP) + np.cosh(-L*afD + Xsrc*afD + Xsrc*afP) + np.cosh(L*afD - Xsrc*afD + Xsrc*afP))/(-afD*gfP*lbdP*np.cosh(-L*afD + Lp*afD + Lp*afP) + afD*gfP*lbdP*np.cosh(L*afD - Lp*afD + Lp*afP) - afD*lbdP*np.sinh(-L*afD + Lp*afD + Lp*afP) + afD*lbdP*np.sinh(L*afD - Lp*afD + Lp*afP) + afP*gfP*lbdD*np.cosh(-L*afD + Lp*afP + Xsrc*afD) + afP*gfP*lbdD*np.cosh(L*afD + Lp*afP - Xsrc*afD) + afP*lbdD*np.sinh(-L*afD + Lp*afP + Xsrc*afD) + afP*lbdD*np.sinh(L*afD + Lp*afP - Xsrc*afD))
@jit
def Cf_coeff(Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP): return If*afD*lbdP*rfP*(gfP*np.cosh(-L*afD + Lp*afD + Xsrc*afP) - gfP*np.cosh(L*afD - Lp*afD + Xsrc*afP) + np.sinh(-L*afD + Lp*afD + Xsrc*afP) - np.sinh(L*afD - Lp*afD + Xsrc*afP))/(afP*(-afD*gfP*lbdP*np.cosh(-L*afD + Lp*afD + Lp*afP) + afD*gfP*lbdP*np.cosh(L*afD - Lp*afD + Lp*afP) - afD*lbdP*np.sinh(-L*afD + Lp*afD + Lp*afP) + afD*lbdP*np.sinh(L*afD - Lp*afD + Lp*afP) + afP*gfP*lbdD*np.cosh(-L*afD + Lp*afP + Xsrc*afD) + afP*gfP*lbdD*np.cosh(L*afD + Lp*afP - Xsrc*afD) + afP*lbdD*np.sinh(-L*afD + Lp*afP + Xsrc*afD) + afP*lbdD*np.sinh(L*afD + Lp*afP - Xsrc*afD)))
@jit
def Df_coeff(Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP): return 2*If*lbdD*rfP*(gfP*np.sinh(Xsrc*afP) + np.cosh(Xsrc*afP))/(-afD*gfP*lbdP*np.cosh(-L*afD + Lp*afD + Lp*afP) + afD*gfP*lbdP*np.cosh(L*afD - Lp*afD + Lp*afP) - afD*lbdP*np.sinh(-L*afD + Lp*afD + Lp*afP) + afD*lbdP*np.sinh(L*afD - Lp*afD + Lp*afP) + afP*gfP*lbdD*np.cosh(-L*afD + Lp*afP + Xsrc*afD) + afP*gfP*lbdD*np.cosh(L*afD + Lp*afP - Xsrc*afD) + afP*lbdD*np.sinh(-L*afD + Lp*afP + Xsrc*afD) + afP*lbdD*np.sinh(L*afD + Lp*afP - Xsrc*afD))
@jit
def dv_X_Xsrc_Lp(X, Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP): return Af_coeff(Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)*(np.cosh(afP*X)+gfP*np.sinh(afP*X)) 
@jit
def dv_Xsrc_X_Lp(X, Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP): return Bf_coeff(Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)*np.cosh(afP*(X-Lp))+Cf_coeff(Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)*np.sinh(afP*(X-Lp)) 
@jit
def dv_Xsrc_Lp_X(X, Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP): return Df_coeff(Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)*np.cosh(afD*(X-L))

@jit
def Ef_coeff(Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP): return 2*If*lbdP*rfD*np.cosh(afD*(L - Xsrc))/(-afD*gfP*lbdP*np.cosh(-L*afD + Lp*afD + Lp*afP) + afD*gfP*lbdP*np.cosh(L*afD - Lp*afD + Lp*afP) - afD*lbdP*np.sinh(-L*afD + Lp*afD + Lp*afP) + afD*lbdP*np.sinh(L*afD - Lp*afD + Lp*afP) + afP*gfP*lbdD*np.cosh(-L*afD + Lp*afD + Lp*afP) + afP*gfP*lbdD*np.cosh(L*afD - Lp*afD + Lp*afP) + afP*lbdD*np.sinh(-L*afD + Lp*afD + Lp*afP) + afP*lbdD*np.sinh(L*afD - Lp*afD + Lp*afP))
@jit
def Ff_coeff(Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP): return If*lbdP*rfD*(gfP*np.sinh(-L*afD + Lp*afP + Xsrc*afD) + gfP*np.sinh(L*afD + Lp*afP - Xsrc*afD) + np.cosh(-L*afD + Lp*afP + Xsrc*afD) + np.cosh(L*afD + Lp*afP - Xsrc*afD))/(-afD*gfP*lbdP*np.cosh(-L*afD + Lp*afD + Lp*afP) + afD*gfP*lbdP*np.cosh(L*afD - Lp*afD + Lp*afP) - afD*lbdP*np.sinh(-L*afD + Lp*afD + Lp*afP) + afD*lbdP*np.sinh(L*afD - Lp*afD + Lp*afP) + afP*gfP*lbdD*np.cosh(-L*afD + Lp*afD + Lp*afP) + afP*gfP*lbdD*np.cosh(L*afD - Lp*afD + Lp*afP) + afP*lbdD*np.sinh(-L*afD + Lp*afD + Lp*afP) + afP*lbdD*np.sinh(L*afD - Lp*afD + Lp*afP))
@jit
def Gf_coeff(Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP): return If*afP*lbdD*rfD*(gfP*np.cosh(-L*afD + Lp*afP + Xsrc*afD) + gfP*np.cosh(L*afD + Lp*afP - Xsrc*afD) + np.sinh(-L*afD + Lp*afP + Xsrc*afD) + np.sinh(L*afD + Lp*afP - Xsrc*afD))/(afD*(-afD*gfP*lbdP*np.cosh(-L*afD + Lp*afD + Lp*afP) + afD*gfP*lbdP*np.cosh(L*afD - Lp*afD + Lp*afP) - afD*lbdP*np.sinh(-L*afD + Lp*afD + Lp*afP) + afD*lbdP*np.sinh(L*afD - Lp*afD + Lp*afP) + afP*gfP*lbdD*np.cosh(-L*afD + Lp*afD + Lp*afP) + afP*gfP*lbdD*np.cosh(L*afD - Lp*afD + Lp*afP) + afP*lbdD*np.sinh(-L*afD + Lp*afD + Lp*afP) + afP*lbdD*np.sinh(L*afD - Lp*afD + Lp*afP)))
@jit
def Hf_coeff(Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP): return If*rfD/(afD*np.sinh(afD*(L - Xsrc)) - (afD**2*lbdP*(gfP*np.sinh(Lp*afP) + np.cosh(Lp*afP))*np.sinh(afD*(Lp - Xsrc))/(afP*lbdD*(gfP*np.cosh(Lp*afP) + np.sinh(Lp*afP))) - afD*np.cosh(afD*(Lp - Xsrc)))*np.cosh(afD*(L - Xsrc))/(afD*lbdP*(gfP*np.sinh(Lp*afP) + np.cosh(Lp*afP))*np.cosh(afD*(Lp - Xsrc))/(afP*lbdD*(gfP*np.cosh(Lp*afP) + np.sinh(Lp*afP))) - np.sinh(afD*(Lp - Xsrc))))
@jit
def dv_X_Lp_Xsrc(X, Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP): return Ef_coeff(Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)*(np.cosh(afP*X)+gfP*np.sinh(afP*X)) 
@jit
def dv_Lp_X_Xsrc(X, Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP): return Ff_coeff(Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)*np.cosh(afD*(X-Lp))+Gf_coeff(Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)*np.sinh(afD*(X-Lp)) 
@jit
def dv_Lp_Xsrc_X(X, Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP): return Hf_coeff(Xsrc, If, afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)*np.cosh(afD*(X-L)) 



muVP = lambda x,Lp,L,lbdD,lbdP,gP,v0D,v0P,V0: ((V0*gP*lbdD*np.cosh((L - Lp)/lbdD)*np.cosh((Lp - x)/lbdP) + V0*gP*lbdP*np.sinh((L - Lp)/lbdD)*np.sinh((Lp - x)/lbdP) + gP*lbdD*v0P*np.cosh(Lp/lbdP)*np.cosh((L - Lp)/lbdD) - gP*lbdD*v0P*np.cosh((L - Lp)/lbdD)*np.cosh((Lp - x)/lbdP) + gP*lbdP*v0D*np.sinh((L - Lp)/lbdD)*np.sinh(x/lbdP) + gP*lbdP*v0P*np.sinh(Lp/lbdP)*np.sinh((L - Lp)/lbdD) - gP*lbdP*v0P*np.sinh((L - Lp)/lbdD)*np.sinh(x/lbdP) - gP*lbdP*v0P*np.sinh((L - Lp)/lbdD)*np.sinh((Lp - x)/lbdP) + lbdD*v0P*np.sinh(Lp/lbdP)*np.cosh((L - Lp)/lbdD) + lbdP*v0D*np.sinh((L - Lp)/lbdD)*np.cosh(x/lbdP) + lbdP*v0P*np.sinh((L - Lp)/lbdD)*np.cosh(Lp/lbdP) - lbdP*v0P*np.sinh((L - Lp)/lbdD)*np.cosh(x/lbdP))/(gP*lbdD*np.cosh(Lp/lbdP)*np.cosh((L - Lp)/lbdD) + gP*lbdP*np.sinh(Lp/lbdP)*np.sinh((L - Lp)/lbdD) + lbdD*np.sinh(Lp/lbdP)*np.cosh((L - Lp)/lbdD) + lbdP*np.sinh((L - Lp)/lbdD)*np.cosh(Lp/lbdP)))
muVD = lambda x,Lp,L,lbdD,lbdP,gP,v0D,v0P,V0: ((lbdD*(gP*(V0 - v0P)*(gP*np.sinh(Lp/lbdP) + np.cosh(Lp/lbdP))*np.cosh(Lp/lbdP) - (gP*np.cosh(Lp/lbdP) + np.sinh(Lp/lbdP))*(gP*(V0 - v0P)*np.sinh(Lp/lbdP) + v0D - v0P))*np.cosh((L - x)/lbdD) + v0D*(lbdD*(gP*np.cosh(Lp/lbdP) + np.sinh(Lp/lbdP))*np.cosh((L - Lp)/lbdD) + lbdP*(gP*np.sinh(Lp/lbdP) + np.cosh(Lp/lbdP))*np.sinh((L - Lp)/lbdD)))/(lbdD*(gP*np.cosh(Lp/lbdP) + np.sinh(Lp/lbdP))*np.cosh((L - Lp)/lbdD) + lbdP*(gP*np.sinh(Lp/lbdP) + np.cosh(Lp/lbdP))*np.sinh((L - Lp)/lbdD)))

# --- dv
dv_xXLp = lambda x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf: (rPf*(gPf*np.sinh(x/lbdPf) + np.cosh(x/lbdPf))*(lbdDf*np.cosh((lbdDf*(Lp - X) - lbdPf*(L - Lp))/(lbdDf*lbdPf)) + lbdDf*np.cosh((lbdDf*(Lp - X) + lbdPf*(L - Lp))/(lbdDf*lbdPf)) - lbdPf*np.cosh((lbdDf*(Lp - X) - lbdPf*(L - Lp))/(lbdDf*lbdPf)) + lbdPf*np.cosh((lbdDf*(Lp - X) + lbdPf*(L - Lp))/(lbdDf*lbdPf)))/(lbdPf*(gPf*lbdDf*np.cosh((Lp*lbdDf - lbdPf*(L - Lp))/(lbdDf*lbdPf)) + gPf*lbdDf*np.cosh((Lp*lbdDf + lbdPf*(L - Lp))/(lbdDf*lbdPf)) - gPf*lbdPf*np.cosh((Lp*lbdDf - lbdPf*(L - Lp))/(lbdDf*lbdPf)) + gPf*lbdPf*np.cosh((Lp*lbdDf + lbdPf*(L - Lp))/(lbdDf*lbdPf)) + lbdDf*np.sinh((Lp*lbdDf - lbdPf*(L - Lp))/(lbdDf*lbdPf)) + lbdDf*np.sinh((Lp*lbdDf + lbdPf*(L - Lp))/(lbdDf*lbdPf)) - lbdPf*np.sinh((Lp*lbdDf - lbdPf*(L - Lp))/(lbdDf*lbdPf)) + lbdPf*np.sinh((Lp*lbdDf + lbdPf*(L - Lp))/(lbdDf*lbdPf)))))
dv_XxLp = lambda x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf: (rPf*(lbdDf*(gPf*np.sinh((X*lbdDf - lbdPf*(L - Lp))/(lbdDf*lbdPf)) + gPf*np.sinh((X*lbdDf + lbdPf*(L - Lp))/(lbdDf*lbdPf)) + np.cosh((X*lbdDf - lbdPf*(L - Lp))/(lbdDf*lbdPf)) + np.cosh((X*lbdDf + lbdPf*(L - Lp))/(lbdDf*lbdPf)))*np.cosh((Lp - x)/lbdPf) - lbdPf*(gPf*np.cosh((X*lbdDf - lbdPf*(L - Lp))/(lbdDf*lbdPf)) - gPf*np.cosh((X*lbdDf + lbdPf*(L - Lp))/(lbdDf*lbdPf)) + np.sinh((X*lbdDf - lbdPf*(L - Lp))/(lbdDf*lbdPf)) - np.sinh((X*lbdDf + lbdPf*(L - Lp))/(lbdDf*lbdPf)))*np.sinh((Lp - x)/lbdPf))/(lbdPf*(gPf*lbdDf*np.cosh((Lp*lbdDf - lbdPf*(L - Lp))/(lbdDf*lbdPf)) + gPf*lbdDf*np.cosh((Lp*lbdDf + lbdPf*(L - Lp))/(lbdDf*lbdPf)) - gPf*lbdPf*np.cosh((Lp*lbdDf - lbdPf*(L - Lp))/(lbdDf*lbdPf)) + gPf*lbdPf*np.cosh((Lp*lbdDf + lbdPf*(L - Lp))/(lbdDf*lbdPf)) + lbdDf*np.sinh((Lp*lbdDf - lbdPf*(L - Lp))/(lbdDf*lbdPf)) + lbdDf*np.sinh((Lp*lbdDf + lbdPf*(L - Lp))/(lbdDf*lbdPf)) - lbdPf*np.sinh((Lp*lbdDf - lbdPf*(L - Lp))/(lbdDf*lbdPf)) + lbdPf*np.sinh((Lp*lbdDf + lbdPf*(L - Lp))/(lbdDf*lbdPf)))))
dv_XLpx = lambda x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf: (lbdDf*rPf*(gPf*np.sinh(X/lbdPf) + np.cosh(X/lbdPf))*np.cosh((L - x)/lbdDf)/(lbdPf*(gPf*lbdDf*np.cosh(Lp/lbdPf)*np.cosh((L - Lp)/lbdDf) + gPf*lbdPf*np.sinh(Lp/lbdPf)*np.sinh((L - Lp)/lbdDf) + lbdDf*np.sinh(Lp/lbdPf)*np.cosh((L - Lp)/lbdDf) + lbdPf*np.sinh((L - Lp)/lbdDf)*np.cosh(Lp/lbdPf))))

dv_xLpX = lambda x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf: (-lbdPf*rDf*(gPf*np.sinh(x/lbdPf) + np.cosh(x/lbdPf))*np.exp(-(L - X)/lbdDf)*np.cosh((L - X)/lbdDf)/(lbdDf*(lbdDf*(gPf*np.cosh(Lp/lbdPf) + np.sinh(Lp/lbdPf))*np.sinh((Lp - X)/lbdDf) - lbdPf*(gPf*np.sinh(Lp/lbdPf) + np.cosh(Lp/lbdPf))*np.cosh((Lp - X)/lbdDf))))
dv_LpxX = lambda x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf: (rDf*(lbdDf*(gPf*np.cosh(Lp/lbdPf) + np.sinh(Lp/lbdPf))*np.sinh((Lp - x)/lbdDf) - lbdPf*(gPf*np.sinh(Lp/lbdPf) + np.cosh(Lp/lbdPf))*np.cosh((Lp - x)/lbdDf))*np.exp(-(L - X)/lbdDf)*np.cosh((L - X)/lbdDf)/(lbdDf*(lbdDf*(gPf*np.cosh(Lp/lbdPf) + np.sinh(Lp/lbdPf))*np.sinh((Lp - X)/lbdDf) - lbdPf*(gPf*np.sinh(Lp/lbdPf) + np.cosh(Lp/lbdPf))*np.cosh((Lp - X)/lbdDf))))
dv_LpXx = lambda x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf: (rDf*np.exp(-(L - X)/lbdDf)*np.cosh((L - x)/lbdDf)/lbdDf)

def rescale_x(x, EqCylinder):
    C = EqCylinder[EqCylinder<=x]
    factor = np.power(2., 1./3.*np.arange(len(C)))
    return np.sum(np.diff(C)*factor[:-1])+(x-C[-1])*factor[-1]

def lbd(x, l, lp, B, lbdP, lbdD):
    # specific to evenly space branches !! (see older implementation with EqCylinder for more general implement.)
    branch_length = l/B # length of one branch !
    reduction_factor =  2**(-1./3.*np.intp(x/branch_length))
    new_lbd = (lbdP+(lbdD-lbdP)*.5*(np.sign(x+1e-9-lp)+1))
    return new_lbd*reduction_factor
    
def rescale2_x(x, l, lp, B, lbdP, lbdD):
    # specific to evenly space branches !! (see older implementation with EqCylinder for more general implement.)
    # EqCylinder = np.sort(np.concatenate([np.linspace(0, l, B+1), [lp]]))
    EqCylinder = np.linspace(0, l, B+1)
    C = EqCylinder[EqCylinder<=x]
    return np.sum(np.diff(C)/lbd(C, l, lp, B, lbdP, lbdD)[:-1])+(x-C[-1])/lbd(C[-1], l, lp, B, lbdP, lbdD)


def stat_pot_function(x, shtn_input, EqCylinder, soma, stick, Params):

    params_for_cable_theory(stick, Params)

    Ls, Ds, l, D, lp, Rm, Cm,\
        El, Ee, Ei, rm, cm, ri = ball_and_stick_params(soma, stick, Params)

    Gi_soma, ge_prox, gi_prox, ge_dist, gi_dist = \
                calculate_mean_conductances(shtn_input,\
                                            soma, stick, Params)
    tauS, tauP, lbdP, tauD, lbdD = \
            ball_and_stick_constants(shtn_input, soma, stick, Params)

    # proximal params
    v0P = (El+rm*ge_prox*Ee+rm*gi_prox*Ei)/(1+rm*ge_prox+rm*gi_prox)
    gP = Cm*ri*lbdP/tauS
    # distal params
    v0D = (El+rm*ge_dist*Ee+rm*gi_dist*Ei)/(1+rm*ge_dist+rm*gi_dist)
    # somatic params
    V0 = (El+Rm*Gi_soma*Ei)/(1+Rm*Gi_soma)

    X = np.array([rescale2_x(xx, l, lp, Params['B'], lbdP, lbdD) for xx in x])
    Lp, L = rescale2_x(lp, l, lp, Params['B'], lbdP, lbdD), rescale2_x(l, l, lp, Params['B'], lbdP, lbdD)

    return np.array([muV_prox(XX, Lp, L, lbdD, lbdP, gP, v0D, v0P, V0) if XX<Lp\
                     else muV_dist(XX, Lp, L, lbdD, lbdP, gP, v0D, v0P, V0) for XX in X])



@jit
def exp_FT(f, Q, Tsyn, t0=0):
    return Q*np.exp(-1j*2*np.pi*t0*f)/(1j*2*np.pi*f+1./Tsyn)

@jit
def exp_FT_mod(f, Q, Tsyn):
    return Q**2/((2*np.pi*f)**2+(1./Tsyn)**2)

@jit
def split_root_square_of_imaginary(f, tau):
    # returns the ral and imaginary part of Sqrt(1+2*Pi*f*tau)
    af = np.sqrt((np.sqrt(1+(2*np.pi*f*tau)**2)+1)/2)
    bf = np.sqrt((np.sqrt(1+(2*np.pi*f*tau)**2)-1)/2)
    return af, bf

@jit
def psp_norm_square_integral_per_dend_synapse_type(x, X, f, Gf2,\
                            Erev, shtn_input, EqCylinder,\
                            soma, stick, params,
                            precision=1e2):
    
    # muV for mean driving force
    muV_X = stat_pot_function([X], shtn_input, EqCylinder,\
                              soma, stick, params)[0]
    
    # model parameters
    Ls, Ds, l, D, lp, Rm, Cm,\
        El, Ee, Ei, rm, cm, ri = ball_and_stick_params(soma, stick, params)
    # activity dependent parameters
    tauS, tauP, lbdP, tauD, lbdD = \
            ball_and_stick_constants(shtn_input, soma, stick, params)

    # proximal params
    afP = np.sqrt(1+2*1j*np.pi*f*tauP)
    gfP = lbdP*Cm*ri*(1+2*1j*np.pi*f*tauS)/tauS
    rfP = tauP/cm/lbdP/(1+2*1j*np.pi*f*tauP)*afP

    # distal params
    afD = np.sqrt(1+2*1j*np.pi*f*tauD)
    rfD = tauD/cm/lbdD/(1+2*1j*np.pi*f*tauD)*afD

    # ball and tree rescaling
    Lp = rescale2_x(lp, l, lp, params['B'], lbdP, lbdD)
    L = rescale2_x(l, l, lp, params['B'], lbdP, lbdD)
    Xsrc = rescale2_x(X, l, lp, params['B'], lbdP, lbdD)
    Xdest = rescale2_x(x, l, lp, params['B'], lbdP, lbdD)

    # PSP with unitary current input
    if Xsrc<=Lp:
        if Xdest<=Xsrc:
            # PSP = dv_xXLp(x, X, Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)
            PSP = dv_X_Xsrc_Lp(Xdest, Xsrc, 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)
        elif Xdest>Xsrc and Xdest<=Lp:
            # PSP = dv_XxLp(x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)
            PSP = dv_Xsrc_X_Lp(Xdest, Xsrc, 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)
        elif Xdest>Xsrc and Xdest>Lp:
            # PSP = dv_XLpx(x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)
            PSP = dv_Xsrc_Lp_X(Xdest, Xsrc, 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)
    elif Xsrc>Lp:
        if Xdest<=Lp:
            # PSP = dv_xLpX(x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)
            PSP = dv_X_Lp_Xsrc(Xdest, Xsrc, 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)
        elif Xdest>Lp and Xdest<=Xsrc:
            # PSP = dv_LpxX(x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)
            PSP = dv_Lp_X_Xsrc(Xdest, Xsrc, 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)
        elif Xdest>Xsrc:
            # PSP = dv_LpXx(x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)
            PSP = dv_Lp_Xsrc_X(Xdest, Xsrc, 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)

    return Gf2*np.abs(PSP)**2*(Erev-muV_X)**2

# @jit
# def psp_norm_square_integral_per_dend_synapse_type(x, X, f, Gf2,\
#                             Erev, shtn_input, EqCylinder,\
#                             soma, stick, params,
#                             precision=1e2):

#     Ls, Ds, L, D, Lp, Rm, Cm,\
#         El, Ee, Ei, rm, cm, ri = ball_and_stick_params(soma, stick, params)

#     tauS, tauP, lbdP, tauD, lbdD = \
#             ball_and_stick_constants(shtn_input, soma, stick, params)

#     # proximal params
#     lbdPf = lbdP/np.sqrt(1+2*1j*np.pi*f*tauP)
#     gPf = lbdPf*Cm*ri*(1+2*1j*np.pi*f*tauS)/tauS
#     rPf = tauP/cm/(1+2*1j*np.pi*f*tauP)

#     # distal params
#     lbdDf = lbdD/np.sqrt(1+2*1j*np.pi*f*tauD)
#     rDf = tauD/cm/(1+2*1j*np.pi*f*tauD)

#     # muV for mean driving force
#     muV_X = stat_pot_function([X], shtn_input, EqCylinder,\
#                               soma, stick, params)[0]

#     # ball and tree rescaling
#     Lp, L = rescale_x(Lp, EqCylinder), rescale_x(L, EqCylinder)
#     x, X = rescale_x(x,EqCylinder), rescale_x(X,EqCylinder)

#     # PSP with unitary current input
#     if X<=Lp:
#         if x<=X:
#             PSP = dv_xXLp(x, X, Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)
#         elif x>X and x<=Lp:
#             PSP = dv_XxLp(x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)
#         elif x>X and x>Lp:
#             PSP = dv_XLpx(x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)
#     elif X>Lp:
#         if x<=Lp:
#             PSP = dv_xLpX(x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)
#         elif x>Lp and x<=X:
#             PSP = dv_LpxX(x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)
#         elif x>X:
#             PSP = dv_LpXx(x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)

#     return Gf2*np.abs(PSP)**2*(Erev-muV_X)**2


# @jit
def get_the_theoretical_sV_and_Tv(shtn_input, EqCylinder,\
                                  f, x, params, soma, stick,\
                                  precision=50):
    Pv = np.zeros((len(x), len(f))) # power spectral density of Vm for each position

    Source_Array = np.linspace(x.min(), x.max(), precision+1)
    Source_Array = .5*(Source_Array[1:]+Source_Array[:-1]) # discarding the soma, treated below
    DX = Source_Array[1]-Source_Array[0]
    Branch_weights = 0*Source_Array # initialized t0 0 !

    synchrony = shtn_input['synchrony']
    synch_factor = 1.+3.*synchrony # factor for the PSP event
    synch_dividor = 1.+synchrony # dividor for the real frequency
    # rational: new_freq = freq/(1.+synchrony)
    # we generate spikes with frequency : new_freq
    # then we duplicate each spike with a probability : 'synchrony'
    # so we get independent events (1 spike) with frequency new_freq*(1-synchrony)
    # and we get double events with freq: new_freq*synchrony
    # double envents have a weight (2*PSP)**2 -> 4
    # new_freq*(1-synchrony+4*synchrony)

    
    for b in EqCylinder:
        Branch_weights[np.where(Source_Array>=b)[0]] += 1

    for ix_dest in range(len(x)):

        #### DENDRITIC SYNAPSES
        for ix_source in range(len(Source_Array)): 

            X_source = Source_Array[ix_source]
            if X_source<=stick['L']*params['fraction_for_L_prox']:
                fe, fi = shtn_input['fe_prox'], shtn_input['fi_prox']
                weight_synapse_factor = 1.
                tau_synapse_factor = 1.
            else:
                fe, fi = shtn_input['fe_dist'], shtn_input['fi_dist']
                weight_synapse_factor = params['factor_for_distal_synapses_weight']
                tau_synapse_factor = params['factor_for_distal_synapses_tau']
                
            fe /= synch_dividor
            fi /= synch_dividor
            ## weighting due to branching !
            fe, fi = fe*Branch_weights[ix_source], fi*Branch_weights[ix_source]
            Qe, Qi = weight_synapse_factor*params['Qe']/Branch_weights[ix_source],\
                     weight_synapse_factor*params['Qi']/Branch_weights[ix_source]

            # excitatory synapse at dendrites
            Gf2 = exp_FT_mod(f, Qe, params['Te']*tau_synapse_factor)
            psp2 = psp_norm_square_integral_per_dend_synapse_type(\
                            x[ix_dest], X_source,\
                            f, Gf2, params['Ee'], shtn_input, EqCylinder,\
                            soma, stick, params, precision=precision)
            Pv[ix_dest,:] += np.pi*fe*DX*stick['D']/stick['exc_density']*psp2*synch_factor

            # inhibitory synapse at dendrites
            Gf2 = exp_FT_mod(f, Qi, params['Ti']*tau_synapse_factor)
            psp2 = psp_norm_square_integral_per_dend_synapse_type(\
                            x[ix_dest], X_source,\
                            f, Gf2, params['Ei'], shtn_input, EqCylinder,\
                            soma, stick, params, precision=precision)
            Pv[ix_dest,:] += np.pi*fi*DX*stick['D']/stick['inh_density']*psp2*synch_factor

        # #### SOMATIC SYNAPSES, discret summation, only inhibition, no branch weighting
        fi = shtn_input['fi_soma']
        fi /= synch_dividor
        Gf2 = exp_FT_mod(f, params['Qi'], params['Ti'])
        psp2 = psp_norm_square_integral_per_dend_synapse_type(\
                        x[ix_dest], 0.,\
                        f, Gf2, params['Ei'], shtn_input, EqCylinder,\
                        soma, stick, params, precision=precision)
        Pv[ix_dest,:] += np.pi*fi*soma['L']*soma['D']/soma['inh_density']*psp2*synch_factor

    sV2, Tv = np.zeros(len(x)), np.zeros(len(x))
    for ix in range(len(x)):
        sV2[ix] = 2.*np.trapz(np.abs(Pv[ix,:]), f)
        Tv[ix] = .5*Pv[ix,0]/(2.*np.trapz(np.abs(Pv[ix,:]), f)) # 2 times the integral to have from -infty to +infty (and methods gives [0,+infty])
        
    return np.sqrt(sV2), Tv

@jit
def psp_norm_square_integral_per_dend_synapse_type_at_soma(X, f, Gf2,\
                            Erev, shtn_input, EqCylinder,\
                            soma, stick, params,
                            precision=1e2):

    Ls, Ds, L, D, Lp, Rm, Cm,\
        El, Ee, Ei, rm, cm, ri = ball_and_stick_params(soma, stick, params)

    tauS, tauP, lbdP, tauD, lbdD = \
            ball_and_stick_constants(shtn_input, soma, stick, params)

    # proximal params
    lbdPf = lbdP/np.sqrt(1+2*1j*np.pi*f*tauP)
    gPf = lbdPf*Cm*ri*(1+2*1j*np.pi*f*tauS)/tauS
    rPf = tauP/cm/(1+2*1j*np.pi*f*tauP)

    # distal params
    lbdDf = lbdD/np.sqrt(1+2*1j*np.pi*f*tauD)
    rDf = tauD/cm/(1+2*1j*np.pi*f*tauD)

    # muV for mean driving force
    muV_X = stat_pot_function([X], shtn_input, EqCylinder,\
                              soma, stick, params)[0]

    # ball and tree rescaling
    Lp, L = rescale_x(Lp, EqCylinder), rescale_x(L, EqCylinder)
    X = rescale_x(X,EqCylinder)

    # PSP with unitary current input
    if X<=Lp:
        # to be replaced by simplified expression
        PSP = dv_xXLp(0., X, Lp,L,lbdDf,lbdPf,gPf,rPf,rDf) 
    elif X>Lp:
        # to be replaced by simplified expression
        PSP = dv_xLpX(0.,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)

    return Gf2*np.abs(PSP)**2*(Erev-muV_X)**2

@jit
def get_the_fluct_prop_at_soma(SHTN_INPUT, params, soma, stick,\
                               precision=100, f=rfft.time_to_freq(1000, 1e-4)):


    EqCylinder = np.linspace(0,1,stick['B']+1)*stick['L']
    params_for_cable_theory(stick, params)
    setup_model(EqCylinder, soma, stick, params)
    
    # check if the shtn input is an array
    n = len(SHTN_INPUT['fi_soma'])

    muV, sV, TvN, muGn = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        
    # input resistance at rest
    Rin0 = get_the_input_resistance_at_soma(EqCylinder, soma, stick, params,
                            {'fi_soma':0, 'fe_prox':0,'fi_prox':0,
                             'fe_dist':0,'fi_dist':0, 'synchrony':0.})
    # membrane time constant at rest
    Tm0 = get_membrane_time_constants(EqCylinder, soma, stick, params)

    # then temporal loop
    for i in range(n):
        
        shtn_input = {'fi_soma':SHTN_INPUT['fi_soma'][i], 'fe_prox':SHTN_INPUT['fe_prox'][i],\
                      'fi_prox':SHTN_INPUT['fi_prox'][i], 'fe_dist':SHTN_INPUT['fe_dist'][i],\
                      'fi_dist':SHTN_INPUT['fi_dist'][i], 'synchrony':SHTN_INPUT['synchrony'][i]}

        synchrony = shtn_input['synchrony']
        synch_factor = 1.+3.*synchrony # factor for the PSP event
        synch_dividor = 1.+synchrony # dividor for the real frequency

        Pv = np.zeros(len(f)) # power spectral density of Vm for each position

        Source_Array = np.linspace(0, stick['L'], precision+1)
        Source_Array = .5*(Source_Array[1:]+Source_Array[:-1]) # discarding the soma, treated below
        DX = Source_Array[1]-Source_Array[0]
        Branch_weights = 0*Source_Array # initialized t0 0 !
        for b in EqCylinder:
            Branch_weights[np.where(Source_Array>=b)[0]] += 1


        #### DENDRITIC SYNAPSES
        for ix_source in range(len(Source_Array)): 

            X_source = Source_Array[ix_source]
            if X_source<=stick['L']*params['fraction_for_L_prox']:
                fe, fi = shtn_input['fe_prox'], shtn_input['fi_prox']
                weight_synapse_factor = 1.
                tau_synapse_factor = 1.
            else:
                fe, fi = shtn_input['fe_dist'], shtn_input['fi_dist']
                weight_synapse_factor = params['factor_for_distal_synapses_weight']
                tau_synapse_factor = params['factor_for_distal_synapses_tau']
                
            synchrony = shtn_input['synchrony']
            
            ## weighting due to branching !
            fe, fi = fe*Branch_weights[ix_source], fi*Branch_weights[ix_source]
            fe /= synch_dividor
            fi /= synch_dividor
            Qe, Qi = weight_synapse_factor*params['Qe']/Branch_weights[ix_source],\
                     weight_synapse_factor*params['Qi']/Branch_weights[ix_source]

            # excitatory synapse at dendrites
            Gf2 = exp_FT_mod(f, Qe, params['Te']*tau_synapse_factor)
            psp2 = psp_norm_square_integral_per_dend_synapse_type_at_soma(\
                            X_source,\
                            f, Gf2, params['Ee'], shtn_input, EqCylinder,\
                            soma, stick, params, precision=precision)
            Pv += np.pi*fe*DX*stick['D']/stick['exc_density']*psp2*synch_factor

            # inhibitory synapse at dendrites
            Gf2 = exp_FT_mod(f, Qi, params['Ti']*tau_synapse_factor)
            psp2 = psp_norm_square_integral_per_dend_synapse_type_at_soma(\
                            X_source,\
                            f, Gf2, params['Ei'], shtn_input, EqCylinder,\
                            soma, stick, params, precision=precision)
            Pv += np.pi*fi*DX*stick['D']/stick['inh_density']*psp2*synch_factor

        # #### SOMATIC SYNAPSES, discret summation, only inhibition, no branch weighting
        fi = shtn_input['fi_soma']
        Gf2 = exp_FT_mod(f, params['Qi'], params['Ti'])
        psp2 = psp_norm_square_integral_per_dend_synapse_type_at_soma(\
                        0.,\
                        f, Gf2, params['Ei'], shtn_input, EqCylinder,\
                        soma, stick, params, precision=precision)
        Pv += np.pi*fi*soma['L']*soma['D']/soma['inh_density']*psp2*synch_factor

        Rin = get_the_input_resistance_at_soma(EqCylinder, soma, stick, params,
                                               shtn_input)

        muV[i] = stat_pot_function([0], shtn_input, EqCylinder,\
                                soma, stick, params)[0]

        sV[i] = np.sqrt(2.*np.trapz(np.abs(Pv), f))

        TvN[i] = .5*Pv[0]/(2.*np.trapz(np.abs(Pv), f))/Tm0 # 2 times the integral to have from -infty to +infty (and methods gives [0,+infty])

        muGn[i] = Rin0/Rin

    return muV, sV, TvN, muGn


def get_the_input_and_transfer_resistance(fe, fi, f, x, params, soma, stick):
    Rin, Rtf = np.zeros(len(x)), np.zeros(len(x))
    Garray = calculate_mean_conductances(fe, fi, soma, stick, params)
    
    for ix_dest in range(len(x)):

        Rtf[ix_dest] = dv_kernel_per_I(0., x[ix_dest],
                0., Garray, stick, soma, params) # at 0 frequency
        Rin[ix_dest] = dv_kernel_per_I(x[ix_dest], x[ix_dest],
                0., Garray, stick, soma, params) # at 0 frequency
    return Rin, Rtf


def get_the_input_resistance_at_soma(EqCylinder, soma, stick, params,
                                     shtn_input):

    Ls, Ds, L, D, Lp, Rm, Cm,\
        El, Ee, Ei, rm, cm, ri = ball_and_stick_params(soma, stick, params)

    # impact of the BRANCHING !!
    Lp, L = rescale_x(Lp, EqCylinder), rescale_x(L, EqCylinder)
    
    tauS, tauP, lbdP, tauD, lbdD = \
            ball_and_stick_constants(shtn_input, soma, stick, params)
    f=0
    # proximal params
    lbdPf = lbdP/np.sqrt(1+2*1j*np.pi*f*tauP)
    gPf = lbdPf*Cm*ri*(1+2*1j*np.pi*f*tauS)/tauS
    rPf = tauP/cm/(1+2*1j*np.pi*f*tauP)

    # distal params
    lbdDf = lbdD/np.sqrt(1+2*1j*np.pi*f*tauD)
    rDf = tauD/cm/(1+2*1j*np.pi*f*tauD)

    # PSP with unitary current input
    # input and recording in x=0 
    return np.abs(dv_xXLp(0., 0., Lp, L ,lbdDf,lbdPf,gPf,rPf,rDf))


def get_the_input_impedance_at_soma(f, EqCylinder, soma, stick, params):

    # we remove the prox/dist separation, usefull only when synaptic input !!
    stick = stick.copy()
    
    Ls, Ds, L, D, Lp, Rm, Cm,\
        El, Ee, Ei, rm, cm, ri = ball_and_stick_params(soma, stick, params)
    Lp = L # not need of splitting the tree with respect ot proximal and distal
    # impact of the BRANCHING !!
    Lp, L = rescale_x(Lp, EqCylinder), rescale_x(L, EqCylinder)
    
    # activity set to 0 !!
    shtn_input = {'fi_soma':0, 'fe_prox':0,'fi_prox':0,
                  'fe_dist':0,'fi_dist':0}
    tauS, tauP, lbdP, tauD, lbdD = \
            ball_and_stick_constants(shtn_input, soma, stick, params)

    # proximal params
    lbdPf = lbdP/np.sqrt(1+2*1j*np.pi*f*tauP)
    gPf = lbdPf*Cm*ri*(1+2*1j*np.pi*f*tauS)/tauS
    rPf = tauP/cm/(1+2*1j*np.pi*f*tauP)

    # distal params
    lbdDf = lbdD/np.sqrt(1+2*1j*np.pi*f*tauD)
    rDf = tauD/cm/(1+2*1j*np.pi*f*tauD)

    # PSP with unitary current input
    # input and recording in x=0 
    return dv_xXLp(0., 0., Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)

def get_membrane_time_constants(EqCylinder, soma, stick, params,\
                                f=rfft.time_to_freq(1000, 1e-4)):
    psd = np.abs(get_the_input_impedance_at_soma(f, EqCylinder, soma, stick, params))**2
    return .5*psd[0]/(2.*np.trapz(np.abs(psd), f))


def find_balance_at_soma(Fe_prox, Fe_dist, params, soma, stick,\
                         balance=-60e-3, precision=1e2):

    EqCylinder = np.linspace(0,1,stick['B']+1)*stick['L']
    
    shtn_input = {'fe_prox':Fe_prox, 'fe_dist':Fe_dist}
    FI_prox = np.linspace(Fe_prox/2., 10.*Fe_prox, int(precision))
    FI_dist = np.linspace(Fe_dist/2., 10.*Fe_dist, int(precision))

    muV = 0.*FI_dist
    for i in range(int(precision)):
        shtn_input['fi_prox'], shtn_input['fi_dist']= FI_prox[i], FI_dist[i]
        shtn_input['fi_soma']= FI_prox[i]
        muV[i] = stat_pot_function([0], shtn_input, EqCylinder,\
                                soma, stick, params)[0]
    i0 = np.argmin(np.abs(muV-balance))
    return FI_prox[i0], FI_dist[i0]
