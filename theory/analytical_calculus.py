import numpy as np
import matplotlib
import matplotlib.pylab as plt
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

    L, D, Lp = cable['L'], cable['D'], cable['L_prox']
    Te, Qe = Params['Te'], Params['Qe']
    Ti, Qi = Params['Ti'], Params['Qi']

    ge_prox = Qe*shtn_input['fe_prox']*Te*D*np.pi/cable['exc_density']
    gi_prox = Qi*shtn_input['fi_prox']*Ti*D*np.pi/cable['inh_density']
    ge_dist = Qe*shtn_input['fe_dist']*Te*D*np.pi/cable['exc_density']
    gi_dist = Qi*shtn_input['fi_dist']*Ti*D*np.pi/cable['inh_density']

    # somatic inhibitory conductance
    Ls, Ds = soma['L'], soma['D']
    Kis = np.pi*Ls*Ds/soma['inh_density']
    Gi_soma = Qi*Kis*shtn_input['fi_soma']*Ti
    # print 'Gi soma (nS) : ', 1e9*Gi
    
    return Gi_soma, ge_prox, gi_prox, ge_dist, gi_dist


def ball_and_stick_params(soma, stick, Params):
    Ls, Ds = soma['L'], soma['D']
    L, Lp, D = stick['L'], stick['L_prox'], stick['D']
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

def stat_pot_function(x, shtn_input, EqCylinder, soma, stick, Params):

    params_for_cable_theory(stick, Params)

    Ls, Ds, L, D, Lp, Rm, Cm,\
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
    
    Lp, L = rescale_x(Lp, EqCylinder), rescale_x(L, EqCylinder)
    return np.array([muVP(rescale_x(xx, EqCylinder),Lp,L,lbdD,lbdP,gP,v0D,v0P,V0) if xx<Lp\
        else muVD(rescale_x(xx, EqCylinder),Lp,L,lbdD,lbdP,gP,v0D,v0P,V0) for xx in x])


def exp_FT(f, Q, Tsyn, t0=0):
    return Q*np.exp(-1j*2*np.pi*t0*f)/(1j*2*np.pi*f+1./Tsyn)

def exp_FT_mod(f, Q, Tsyn):
    return Q**2/((2*np.pi*f)**2+(1./Tsyn)**2)

def split_root_square_of_imaginary(f, tau):
    # returns the ral and imaginary part of Sqrt(1+2*Pi*f*tau)
    af = np.sqrt((np.sqrt(1+(2*np.pi*f*tau)**2)+1)/2)
    bf = np.sqrt((np.sqrt(1+(2*np.pi*f*tau)**2)-1)/2)
    return af, bf

from numba import jit

@jit
def psp_norm_square_integral_per_dend_synapse_type(x, X, f, Gf2,\
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
    x, X = rescale_x(x,EqCylinder), rescale_x(X,EqCylinder)

    # PSP with unitary current input
    if X<=Lp:
        if x<=X:
            PSP = dv_xXLp(x, X, Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)
        elif x>X and x<=Lp:
            PSP = dv_XxLp(x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)
        elif x>X and x>Lp:
            PSP = dv_XLpx(x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)
    elif X>Lp:
        if x<=Lp:
            PSP = dv_xLpX(x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)
        elif x>Lp and x<=X:
            PSP = dv_LpxX(x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)
        elif x>X:
            PSP = dv_LpXx(x,X,Lp,L,lbdDf,lbdPf,gPf,rPf,rDf)

    return Gf2*np.abs(PSP)**2*(Erev-muV_X)**2

@jit
def get_the_theoretical_sV_and_Tv(shtn_input, EqCylinder,\
                                  f, x, params, soma, stick,\
                                  precision=50):
    Pv = np.zeros((len(x), len(f))) # power spectral density of Vm for each position

    Source_Array = np.linspace(x.min(), x.max(), precision+1)
    Source_Array = .5*(Source_Array[1:]+Source_Array[:-1]) # discarding the soma, treated below
    DX = Source_Array[1]-Source_Array[0]
    Branch_weights = 0*Source_Array # initialized t0 0 !
    for b in EqCylinder:
        Branch_weights[np.where(Source_Array>=b)[0]] += 1

    for ix_dest in range(len(x)):

        #### DENDRITIC SYNAPSES
        for ix_source in range(len(Source_Array)): 

            X_source = Source_Array[ix_source]
            if X_source<=stick['L_prox']:
                fe, fi = shtn_input['fe_prox'], shtn_input['fi_prox']
            else:
                fe, fi = shtn_input['fe_dist'], shtn_input['fi_dist']
                
            ## weighting due to branching !
            fe, fi = fe*Branch_weights[ix_source], fi*Branch_weights[ix_source]
            Qe, Qi = params['Qe']/Branch_weights[ix_source],\
                     params['Qi']/Branch_weights[ix_source]

            # excitatory synapse at dendrites
            Gf2 = exp_FT_mod(f, Qe, params['Te'])
            psp2 = psp_norm_square_integral_per_dend_synapse_type(\
                            x[ix_dest], X_source,\
                            f, Gf2, params['Ee'], shtn_input, EqCylinder,\
                            soma, stick, params, precision=precision)
            Pv[ix_dest,:] += np.pi*fe*DX*stick['D']/stick['exc_density']*psp2

            # inhibitory synapse at dendrites
            Gf2 = exp_FT_mod(f, Qi, params['Ti'])
            psp2 = psp_norm_square_integral_per_dend_synapse_type(\
                            x[ix_dest], X_source,\
                            f, Gf2, params['Ei'], shtn_input, EqCylinder,\
                            soma, stick, params, precision=precision)
            Pv[ix_dest,:] += np.pi*fi*DX*stick['D']/stick['inh_density']*psp2

        # #### SOMATIC SYNAPSES, discret summation, only inhibition, no branch weighting
        fi = shtn_input['fi_soma']
        Gf2 = exp_FT_mod(f, params['Qi'], params['Ti'])
        psp2 = psp_norm_square_integral_per_dend_synapse_type(\
                        x[ix_dest], 0.,\
                        f, Gf2, params['Ei'], shtn_input, EqCylinder,\
                        soma, stick, params, precision=precision)
        Pv[ix_dest,:] += np.pi*fi*soma['L']*soma['D']/soma['inh_density']*psp2

    sV2, Tv = np.zeros(len(x)), np.zeros(len(x))
    for ix in range(len(x)):
        sV2[ix] = 2.*np.trapz(np.abs(Pv[ix,:]), f)
        Tv[ix] = .5*Pv[ix,0]/(2.*np.trapz(np.abs(Pv[ix,:]), f)) # 2 times the integral to have from -infty to +infty (and methods gives [0,+infty])
        
    return np.sqrt(sV2), Tv


def get_the_input_and_transfer_resistance(fe, fi, f, x, params, soma, stick):
    Rin, Rtf = np.zeros(len(x)), np.zeros(len(x))
    Garray = calculate_mean_conductances(fe, fi, soma, stick, params)
    
    for ix_dest in range(len(x)):

        Rtf[ix_dest] = dv_kernel_per_I(0., x[ix_dest],
                0., Garray, stick, soma, params) # at 0 frequency
        Rin[ix_dest] = dv_kernel_per_I(x[ix_dest], x[ix_dest],
                0., Garray, stick, soma, params) # at 0 frequency
    return Rin, Rtf


def get_the_input_impedance_at_soma(f, EqCylinder, soma, stick, params):

    # we remove the prox/dist separation, usefull only when synaptic input !!
    stick = stick.copy()
    stick['L_prox'], stick['L_dist'] = stick['L'], stick['L']
    
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

