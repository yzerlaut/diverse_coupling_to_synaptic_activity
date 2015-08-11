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


def calculate_mean_conductances(\
                                fi_soma, fe_prox, fi_prox, fe_dist, fi_dist,\
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

    ge_prox = Qe*fe_prox*Te*D*np.pi/cable['exc_density']
    gi_prox = Qi*fi_prox*Ti*D*np.pi/cable['inh_density']
    ge_dist = Qe*fe_dist*Te*D*np.pi/cable['exc_density']
    gi_dist = Qi*fi_dist*Ti*D*np.pi/cable['inh_density']

    # somatic inhibitory conductance
    Ls, Ds = soma['L'], soma['D']
    Kis = np.pi*Ls*Ds/soma['inh_density']
    Gi_soma = Qi*Kis*fi_soma*Ti
    # print 'Gi soma (nS) : ', 1e9*Gi
    
    return Gi_soma, ge_prox, gi_prox, ge_dist, gi_dist

    
def stat_pot_function(x, fi_soma, fe_prox, fi_prox, fe_dist, fi_dist,
                      soma, stick, Params):

    params_for_cable_theory(stick, Params)

    Gi_soma, ge_prox, gi_prox, ge_dist, gi_dist = \
                calculate_mean_conductances(fi_soma, fe_prox, fi_prox,\
                                            fe_dist, fi_dist,\
                                            soma, stick, Params)

    [Ls, Ds] = soma['L'], soma['D']
    [L, Lp, D] = stick['L'], stick['L_prox'], stick['D']
    Gl = np.pi*Ls*Ds*Params['g_pas']
    Cm = np.pi*Ls*Ds*Params['cm']
    [El, Ei, Ee] = Params['El'], Params['Ei'], Params['Ee']
    [rm, cm, ri] = stick['rm'], stick['cm'], stick['ri']

    # proximal params
    tauP = rm*cm/(1+rm*ge_prox+rm*gi_prox)
    lbdP = np.sqrt(rm/ri/(1+rm*ge_prox+rm*gi_prox))
    v0P = (El+rm*ge_prox*Ee+rm*gi_prox*Ei)/(1+rm*ge_prox+rm*gi_prox)

    # distal params
    tauD = rm*cm/(1+rm*ge_dist+rm*gi_dist)
    lbdD = np.sqrt(rm/ri/(1+rm*ge_dist+rm*gi_dist))
    v0D = (El+rm*ge_dist*Ee+rm*gi_dist*Ei)/(1+rm*ge_dist+rm*gi_dist)

    # somatic params
    TauS = Cm/(Gl+Gi_soma)
    V0 = (Gl*El+Gi_soma*Ei)/(Gl+Gi_soma)

    A = Cm*ri*lbdP/TauS # alpha factor in notebook derivation
    B = (np.cosh(Lp/lbdP)+A*np.sinh(Lp/lbdP))/(np.sinh(Lp/lbdP)+A*np.cosh(Lp/lbdP))

    v1D  = v0P-v0D+A/(1.-A)*(1-B)*(v0P-V0)*np.exp(Lp/lbdP)
    v1D /= np.cosh((Lp-L)/lbdD)-lbdP/lbdD*B*np.sinh((Lp-L)/lbdD)

    v1P = v0D-v0P+v1D*np.cosh((Lp-L)/lbdD)
    v2P = lbdP/lbdD*v1D*np.sinh((Lp-L)/lbdD)

    def vP(x): # proximal potential
        return v0P+v1P*np.cosh((x-Lp)/lbdP)+v2P*np.sinh((x-Lp)/lbdP)
    def vD(x): # proximal potential
        return v0D+v1D*np.cosh((x-L)/lbdD)

    return np.array([vP(xx) if xx<Lp else vD(xx) for xx in x])


# def stat_pot_function(x, Garray, soma, stick, Params):
#     ge0, gi0, Gi0 = Garray # mean conductance input
#     [tau, Tau, lbd, v0, V0] = parameters_for_mean(ge0, gi0, Gi0,\
#                         soma, stick, Params)
#     [ri, L] = stick['ri'], stick['L']
#     [Ls, Ds] = soma['L'], soma['D']
#     Cm = np.pi*Ls*Ds*Params['cm']
#     denom = 1/np.tanh(L/lbd)+(Tau)/(ri*Cm*lbd)
#     return v0+(V0-v0)/denom*np.cosh((x-L)/lbd)/np.sinh(L/lbd)

def parameters_for_mean_BRT(fe, fi, cables, Params):

    soma = cables[0]
    [Ls, Ds] = soma['L'], soma['D']
    Gl = np.pi*Ls*Ds*Params['g_pas']
    print 'soma conductac', Gl*1e9
    Cm = np.pi*Ls*Ds*Params['cm']
    [El, Ei, Ee] = Params['El'], Params['Ei'], Params['Ee']
    
    # radial density of excitatory conductance
    Te, Qe = Params['Te'], Params['Qe']
    ge0_rm = Qe*fe*Te/Params['exc_density_dend']/Params['g_pas']
    
    # radial density of inhibitory conductance
    Ti, Qi = Params['Ti'], Params['Qi']
    gi0_rm = Qi*fi*Ti/Params['inh_density_dend']/Params['g_pas']
    
    # radial density of inhibitory conductance at soma
    Gi0_Rm = Qi*fi*Ti/Params['inh_density_soma']/Params['g_pas']
    
    print 'Conductances ratio :', Gi0_Rm, gi0_rm, ge0_rm

    
    # potentials
    v0 = (El+ge0_rm*Ee+gi0_rm*Ei)/(1+ge0_rm+gi0_rm)
    V0 = (El+Gi0_Rm*Ei)/(1+Gi0_Rm)

    print ' potentials :', 1e3*v0, 1e3*V0
    # time constants (doesn't depends on the diameter)
    Tau = Params['cm']/Params['g_pas']/(1+Gi0_Rm)
    tau = Params['cm']/Params['g_pas']/(1+ge0_rm+gi0_rm)

    # now the different lambda, as a power of the first lbd=lbd0
    LBD, LL, Lmax = [], [], 0
    D = cables[1]['D']
    lbd0 = np.sqrt(D/4/Params['g_pas']/Params['Ra']/(1+ge0_rm+gi0_rm))
    for i in range(1, len(cables)): # discard soma
        lbd = lbd0*2**(-(i-1)/3.)
        LBD.append(lbd)
        LL.append(cables[i]['L'])
        Lmax += cables[i]['L']/lbd
    return [tau, Tau, np.array(LBD), np.array(LL), Lmax, v0, V0]

def rescale_x_BRT(x, LL, LBD):
    Lsum = np.cumsum(np.concatenate([[0],LL]))
    i, X = 1, 0
    while (x>Lsum[i]) :
        X+=LL[i-1]/LBD[i-1]
        i+=1
    return X+(x-Lsum[i-1])/LBD[i-1]

    
def stat_pot_function_BRT(x, fe, fi, cables, Params):
    
    soma = cables[0]
    [Ls, Ds] = soma['L'], soma['D']
    Cm = np.pi*Ls*Ds*Params['cm']
    
    tau, Tau, LBD, LL, Lmax, v0, V0 = \
      parameters_for_mean_BRT(fe, fi, cables, Params)

    ri = Params['Ra']/(np.pi*cables[1]['D']**2/4) # [O/m]
    denom = 1/np.tanh(Lmax)+(Tau)/(ri*Cm*LBD[0])
    X = np.array([rescale_x_BRT(xx, LL, LBD) for xx in x])
    print X.max(), Lmax
    return v0+(V0-v0)/denom*np.cosh(X-Lmax)/np.sinh(Lmax)


def dv_kernel_per_I(x, X, f, Garray, stick, soma, Params):
    [ri, L, D, cm] = stick['ri'], stick['L'], stick['D'], stick['cm']
    [Ls, Ds] = soma['L'], soma['D']
    Cm = np.pi*Ls*Ds*Params['cm']
    ge0, gi0, Gi0 = Garray # mean conductance input
    [tau, Tau, lbd, v0, V0] = parameters_for_mean(ge0, gi0, Gi0,\
                                    soma, stick, Params)
    Af = 1./(1+2*1j*np.pi*f*tau) # unitary input for the kernel
    lbdF = lbd/np.sqrt(1+2*1j*np.pi*f*tau)
    Bf = lbdF*Cm*ri*(1+2*1j*np.pi*f*Tau)/Tau
    factor = Af*tau/cm/lbdF/( np.sinh(L/lbdF) + Bf*np.cosh(L/lbdF) )
    if x<X:
        func = (np.cosh(x/lbdF) + Bf*np.sinh(x/lbdF))*np.cosh((X-L)/lbdF)
    else:
        func = (np.cosh(X/lbdF) + Bf*np.sinh(X/lbdF))*np.cosh((x-L)/lbdF)
    return func*factor

def expr0(U, B, af, bf):      
    return np.abs( np.sinh(U*(af+1j*bf)) + B*np.cosh(U*(af+1j*bf)) )**2

def expr1(U, af, bf):
    return np.abs(np.cosh(U*(af+1j*bf)))**2
    # return .5*(np.cosh(2*af*U)+np.cos(2*bf*U))
    
def expr2(U, B, af, bf):      
    return np.abs( np.cosh(U*(af+1j*bf)) + B*np.sinh(U*(af+1j*bf)) )**2


def dv_kernel_per_I_X_x2(x, X, f, Garray, Erev, muV,
                    stick, soma, Params):
    """
    Kernel when X (the input location) is greater than the
    location of the destination
    e.g. to be used only for the soma !!
    """
    [ri, L, D, cm] = stick['ri'], stick['L'], stick['D'], stick['cm']
    [Ls, Ds] = soma['L'], soma['D']
    Cm = np.pi*Ls*Ds*Params['cm']
    ge0, gi0, Gi0 = Garray # mean conductance input
    [tau, Tau, lbd, v0, V0] = parameters_for_mean(ge0, gi0, Gi0,\
                                    soma, stick, Params)
    Af2 = 1./(1+(2*np.pi*f*tau)**2) # unitary input for the kernel
    af, bf = split_root_square_of_imaginary(f, tau)
    Bf = lbd*Cm*ri*(1+2*1j*np.pi*f*Tau)/Tau/(af+1j*bf)
    factor = tau**2/cm**2/lbd**2*(af**2+bf**2)/\
      expr0(L/lbd, Bf, af, bf)
    ff = expr2(X/lbd, Bf, af, bf)
    gg = expr1((x-L)/lbd, af, bf)
    return Af2*factor*ff*gg*(Erev-muV)**2


def dv_kernel_per_I_x_X2(x, X, f, Garray, Erev, muV,
                    stick, soma, Params):
    """
    Kernel when X (the input location) is smaller than the
    location of the destination
    """
    [ri, L, D, cm] = stick['ri'], stick['L'], stick['D'], stick['cm']
    [Ls, Ds] = soma['L'], soma['D']
    Cm = np.pi*Ls*Ds*Params['cm']
    ge0, gi0, Gi0 = Garray # mean conductance input
    [tau, Tau, lbd, v0, V0] = parameters_for_mean(ge0, gi0, Gi0,\
                                    soma, stick, Params)
    Af2 = 1./(1+(2*np.pi*f*tau)**2) # unitary input for the kernel
    af, bf = split_root_square_of_imaginary(f, tau)
    Bf = lbd*Cm*ri*(1+2*1j*np.pi*f*Tau)/Tau/(af+1j*bf)
    factor = tau**2/cm**2/lbd**2*(af**2+bf**2)/\
      expr0(L/lbd, Bf, af, bf)
    ff = expr2(x/lbd, Bf, af, bf)
    gg = expr1((X-L)/lbd, af, bf)
    return Af2*factor*ff*gg*(Erev-muV)**2


def exp_FT(f, Q, Tsyn, t0=0):
    return Q*np.exp(-1j*2*np.pi*t0*f)/(1j*2*np.pi*f+1./Tsyn)

def exp_FT_mod(f, Q, Tsyn):
    return Q**2/((2*np.pi*f)**2+(1./Tsyn)**2)

def split_root_square_of_imaginary(f, tau):
    # returns the ral and imaginary part of Sqrt(1+2*Pi*f*tau)
    af = np.sqrt((np.sqrt(1+(2*np.pi*f*tau)**2)+1)/2)
    bf = np.sqrt((np.sqrt(1+(2*np.pi*f*tau)**2)-1)/2)
    return af, bf

def psp_0_freq_per_dend_synapse_type(x, X, Gf,\
                            Erev, Garray,\
                            soma, stick, params,
                            precision=1e2):
    [ri, L, D, cm] = stick['ri'], stick['L'], stick['D'], stick['cm']
    [Ls, Ds] = soma['L'], soma['D']
    Cm = np.pi*Ls*Ds*params['cm']
    ge0, gi0, Gi0 = Garray # mean conductance input
    [tau, Tau, lbd, v0, V0] = parameters_for_mean(ge0, gi0, Gi0,\
                                    soma, stick, params)
    v1 = (V0-v0)/( 1/np.tanh(L/lbd)+(Tau)/(ri*Cm*lbd) )/np.sinh(L/lbd)
    B = lbd*Cm*ri/Tau
    
    # unitary input for the kernel
    factor = ( (Erev-v0-v1*np.cosh((X-L)/lbd)) * tau/cm/lbd )/( np.sinh(L/lbd) + B*np.cosh(L/lbd) )
    
    if x<X:
        func = (np.cosh(x/lbd) + B*np.sinh(x/lbd))*np.cosh((X-L)/lbd)
    else:
        func = (np.cosh(X/lbd) + B*np.sinh(X/lbd))*np.cosh((x-L)/lbd)
    return np.abs(Gf*func*factor)


def get_fourier_transform_integral(x, X, f, Gf2, L, lbd, tau, Tau, Cm, ri):
    """
    crucial function here, this would be nice to have analytically
    """
    af, bf = split_root_square_of_imaginary(f, tau)
    Bf = lbd*Cm*ri*(1+2*1j*np.pi*f*Tau)/Tau/(af+1j*bf)

    if X<x:
        A = expr2(X/lbd, Bf, af, bf)
        B = expr1((x-L)/lbd, af, bf)
    else:
        A = expr2(x/lbd, Bf, af, bf)
        B = expr1((X-L)/lbd, af, bf)
    Denom = expr0(L/lbd, Bf, af, bf)*np.sqrt(1.+(2.*np.pi*f*tau)**2)
    return np.trapz(Gf2*A*B/Denom, f)

def psp_norm_square_integral_per_dend_synapse_type(x, X, f, Gf2,\
                            Erev, Garray,\
                            soma, stick, params,
                            precision=1e2):
    [ri, L, D, cm] = stick['ri'], stick['L'], stick['D'], stick['cm']
    [Ls, Ds] = soma['L'], soma['D']
    Cm = np.pi*Ls*Ds*params['cm']
    ge0, gi0, Gi0 = Garray # mean conductance input
    [tau, Tau, lbd, v0, V0] = parameters_for_mean(ge0, gi0, Gi0,\
                                    soma, stick, params)
    v1 = (V0-v0)/( 1/np.tanh(L/lbd)+(Tau)/(ri*Cm*lbd) )/np.sinh(L/lbd)
    
    # unitary input for the kernel
    factor = ( (Erev-v0-v1*np.cosh((X-L)/lbd)) * tau/cm/lbd )**2
    
    return factor*get_fourier_transform_integral(x, X, f, Gf2, L, lbd, tau, Tau, Cm, ri)

def get_the_theoretical_sV_and_Tv(FE, FI, f, x, params, soma, stick,\
                                  precision=50):
    sv2 = np.zeros(len(x))
    Tv = np.zeros(len(x))
    Garray = calculate_mean_conductances(FE, FI, soma, stick, params)
    muV = stat_pot_function(x, Garray, soma, stick, params)

    Source_Array = np.linspace(0, stick['L'], precision+1)
    Source_Array = .5*(Source_Array[1:]+Source_Array[:-1])

    DX = Source_Array[1]-Source_Array[0] # space interval
    norm_for_Tv = 0*Tv
    for ix_dest in range(len(x)):

        #### DENDRITIC SYNAPSES
        for ix_source in range(len(Source_Array)): # less intervals than points

            X_source = Source_Array[ix_source]
            fe = FE[1][ix_source]
            fi = FI[1][ix_source]

            # excitatory synapse at dendrites
            Gf2 = exp_FT_mod(f, params['Qe'], params['Te'])
            psp2 = psp_norm_square_integral_per_dend_synapse_type(\
                            x[ix_dest], X_source,\
                            f, Gf2, params['Ee'],\
                            Garray, soma, stick, params,
                            precision=precision)
            psp0 = psp_0_freq_per_dend_synapse_type(\
                            x[ix_dest], X_source,\
                            params['Qe']*params['Te'], params['Ee'],\
                            Garray, soma, stick, params,
                            precision=precision)
            sv2[ix_dest] += 2.*np.pi*fe*DX*stick['D']/stick['exc_density']*psp2
            Tv[ix_dest] += 2.*np.pi*fe*DX*stick['D']/stick['exc_density']*psp0**3/4./psp2
            norm_for_Tv[ix_dest] += 2.*np.pi*fe*DX*stick['D']/stick['exc_density']*psp0

            # inhibitory synapse at dendrites
            Gf2 = exp_FT_mod(f, params['Qi'], params['Ti'])
            psp2 = psp_norm_square_integral_per_dend_synapse_type(\
                            x[ix_dest], X_source,\
                            f, Gf2, params['Ei'],\
                            Garray, soma, stick, params,
                            precision=precision)
            psp0 = psp_0_freq_per_dend_synapse_type(\
                            x[ix_dest], X_source,\
                            params['Qi']*params['Ti'], params['Ei'],\
                            Garray, soma, stick, params,
                            precision=precision)
            sv2[ix_dest] += 2.*np.pi*fi*DX*stick['D']/stick['inh_density']*psp2
            Tv[ix_dest] += 2.*np.pi*fi*DX*stick['D']/stick['inh_density']*psp0**3/4./psp2
            norm_for_Tv[ix_dest] += 2.*np.pi*fi*DX*stick['D']/stick['inh_density']*psp0

        #### SOMATIC SYNAPSES, discret summation
        fe, fi = FE[0][0], FI[0][0]

        # inhibitory synapse at soma
        Gf2 = exp_FT_mod(f, params['Qi'], params['Ti'])
        psp2 = psp_norm_square_integral_per_dend_synapse_type(\
                        x[ix_dest], 0.,\
                        f, Gf2, params['Ei'],\
                        Garray, soma, stick, params,
                        precision=precision)
        psp0 = psp_0_freq_per_dend_synapse_type(\
                        x[ix_dest], 0.,\
                        params['Qi']*params['Ti'], params['Ei'],\
                        Garray, soma, stick, params,
                        precision=precision)
        sv2[ix_dest] += 2.*np.pi*fi*soma['L']*soma['D']/soma['inh_density']*psp2
        Tv[ix_dest] += 2.*np.pi*fi*soma['L']*soma['D']/soma['inh_density']*psp0**3/4./psp2
        norm_for_Tv[ix_dest] += 2.*np.pi*fi*soma['L']*soma['D']/soma['inh_density']*psp0
        
    return np.sqrt(sv2), Tv/norm_for_Tv


def get_the_input_and_transfer_resistance(fe, fi, f, x, params, soma, stick):
    Rin, Rtf = np.zeros(len(x)), np.zeros(len(x))
    Garray = calculate_mean_conductances(fe, fi, soma, stick, params)
    
    for ix_dest in range(len(x)):

        Rtf[ix_dest] = dv_kernel_per_I(0., x[ix_dest],
                0., Garray, stick, soma, params) # at 0 frequency
        Rin[ix_dest] = dv_kernel_per_I(x[ix_dest], x[ix_dest],
                0., Garray, stick, soma, params) # at 0 frequency
    return Rin, Rtf

