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
    [Ls, Ds] = soma['L'], soma['D']
    [L, Lp, D] = stick['L'], stick['L_prox'], stick['D']
    Gl = np.pi*Ls*Ds*Params['g_pas']
    Cm = np.pi*Ls*Ds*Params['cm']
    [El, Ei, Ee] = Params['El'], Params['Ei'], Params['Ee']
    [rm, cm, ri] = stick['rm'], stick['cm'], stick['ri']
    return Ls, Ds, L, D, Lp, Gl, Cm, El, Ee, Ei, rm, cm, ri

def ball_and_stick_constants(shtn_input, soma, stick, Params):
    Ls, Ds, L, D, Lp, Gl, Cm, El, Ee, Ei, rm, cm, ri = \
                    ball_and_stick_params(soma, stick, Params)
    Gi_soma, ge_prox, gi_prox, ge_dist, gi_dist = \
                calculate_mean_conductances(shtn_input,\
                                            soma, stick, Params)
    tauP = rm*cm/(1+rm*ge_prox+rm*gi_prox)
    lbdP = np.sqrt(rm/ri/(1+rm*ge_prox+rm*gi_prox))
    tauD = rm*cm/(1+rm*ge_dist+rm*gi_dist)
    lbdD = np.sqrt(rm/ri/(1+rm*ge_dist+rm*gi_dist))
    tauS = Cm/(Gl+Gi_soma)
    return tauS, tauP, lbdP, tauD, lbdD
    
def stat_pot_function(x, shtn_input, soma, stick, Params):

    params_for_cable_theory(stick, Params)

    Ls, Ds, L, D, Lp, Gl, Cm,\
        El, Ee, Ei, rm, cm, ri = ball_and_stick_params(soma, stick, Params)

    Gi_soma, ge_prox, gi_prox, ge_dist, gi_dist = \
                calculate_mean_conductances(shtn_input,\
                                            soma, stick, Params)

    tauS, tauP, lbdP, tauD, lbdD = \
            ball_and_stick_constants(shtn_input, soma, stick, Params)

    # proximal params
    v0P = (El+rm*ge_prox*Ee+rm*gi_prox*Ei)/(1+rm*ge_prox+rm*gi_prox)

    # distal params
    v0D = (El+rm*ge_dist*Ee+rm*gi_dist*Ei)/(1+rm*ge_dist+rm*gi_dist)

    # somatic params
    V0 = (Gl*El+Gi_soma*Ei)/(Gl+Gi_soma)

    A = Cm*ri*lbdP/tauS # alpha factor in notebook derivation
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

# def expr0(U, B, af, bf):      
#     return np.abs( np.sinh(U*(af+1j*bf)) + B*np.cosh(U*(af+1j*bf)) )**2

# def expr1(U, af, bf):
#     return np.abs(np.cosh(U*(af+1j*bf)))**2
    
# def expr2(U, B, af, bf):      
#     return np.abs( np.cosh(U*(af+1j*bf)) + B*np.sinh(U*(af+1j*bf)) )**2

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


# def get_fourier_transform_integral(x, X, f, Gf2, L, lbd, tau, Tau, Cm, ri):
#     """
#     crucial function here, this would be nice to have analytically
#     """
#     af, bf = split_root_square_of_imaginary(f, tau)
#     Bf = lbd*Cm*ri*(1+2*1j*np.pi*f*Tau)/Tau/(af+1j*bf)

#     if X<x:
#         A = expr2(X/lbd, Bf, af, bf)
#         B = expr1((x-L)/lbd, af, bf)
#     else:
#         A = expr2(x/lbd, Bf, af, bf)
#         B = expr1((X-L)/lbd, af, bf)
#     Denom = expr0(L/lbd, Bf, af, bf)*np.sqrt(1.+(2.*np.pi*f*tau)**2)
#     return np.trapz(Gf2*A*B/Denom, f)

def psp_norm_square_integral_per_dend_synapse_type(x, X, f, Gf2,\
                            Erev, shtn_input,\
                            soma, stick, params,
                            precision=1e2):
    Ls, Ds, L, D, Lp, Gl, Cm,\
        El, Ee, Ei, rm, cm, ri = ball_and_stick_params(soma, stick, params)

    tauS, tauP, lbdP, tauD, lbdD = \
            ball_and_stick_constants(shtn_input, soma, stick, params)

    aPf, bPf = split_root_square_of_imaginary(f, tauP)
    lbdPf = lbdP/(aPf+1j*bPf)
    aDf, bDf = split_root_square_of_imaginary(f, tauD)
    lbdDf = lbdD/(aDf+1j*bDf)
    
    gPf = lbdP*Cm*ri*(1+2.*1j*np.pi*f*tauS)/tauS/(aPf+1j*bPf)

    def A(X, I): 
        return -I*lbdDf*rm*(gPf + np.exp(2.*X/lbdPf))*np.exp((Lp - X)/lbdPf)/(lbdPf*(-gPf*lbdDf*np.cosh((L - Lp)/lbdDf) + gPf*lbdPf*np.sinh((L - Lp)/lbdDf) + lbdDf*np.exp(2.*Lp/lbdPf)*np.cosh((L - Lp)/lbdDf) + lbdPf*np.exp(2.*Lp/lbdPf)*np.sinh((L - Lp)/lbdDf)))

    def B(X, I): 
        return -I*rm*(lbdDf*np.exp(2.*Lp/lbdPf)*np.cosh((L - Lp)/lbdDf) + lbdDf*np.exp(2.*X/lbdPf)*np.cosh((L - Lp)/lbdDf) + lbdPf*np.exp(2.*Lp/lbdPf)*np.sinh((L - Lp)/lbdDf) - lbdPf*np.exp(2.*X/lbdPf)*np.sinh((L - Lp)/lbdDf))*np.exp(-X/lbdPf)/(2.*lbdPf*(-gPf*lbdDf*np.cosh((L - Lp)/lbdDf) + gPf*lbdPf*np.sinh((L - Lp)/lbdDf) + lbdDf*np.exp(2.*Lp/lbdPf)*np.cosh((L - Lp)/lbdDf) + lbdPf*np.exp(2.*Lp/lbdPf)*np.sinh((L - Lp)/lbdDf)))

    def C(X, I): 
        return -I*rm*(gPf*lbdDf*np.cosh((L - Lp)/lbdDf) - gPf*lbdPf*np.sinh((L - Lp)/lbdDf) + lbdDf*np.exp(2.*X/lbdPf)*np.cosh((L - Lp)/lbdDf) - lbdPf*np.exp(2.*X/lbdPf)*np.sinh((L - Lp)/lbdDf))*np.exp(-X/lbdPf)/(2.*lbdPf*(-gPf*lbdDf*np.cosh((L - Lp)/lbdDf) + gPf*lbdPf*np.sinh((L - Lp)/lbdDf) + lbdDf*np.exp(2.*Lp/lbdPf)*np.cosh((L - Lp)/lbdDf) + lbdPf*np.exp(2.*Lp/lbdPf)*np.sinh((L - Lp)/lbdDf)))

    def D(X, I): 
        return -I*rm*(gPf*lbdDf*np.cosh((L - Lp)/lbdDf) + gPf*lbdPf*np.sinh((L - Lp)/lbdDf) + lbdDf*np.exp(2.*X/lbdPf)*np.cosh((L - Lp)/lbdDf) + lbdPf*np.exp(2.*X/lbdPf)*np.sinh((L - Lp)/lbdDf))*np.exp((2.*Lp - X)/lbdPf)/(2.*lbdPf*(-gPf*lbdDf*np.cosh((L - Lp)/lbdDf) + gPf*lbdPf*np.sinh((L - Lp)/lbdDf) + lbdDf*np.exp(2.*Lp/lbdPf)*np.cosh((L - Lp)/lbdDf) + lbdPf*np.exp(2.*Lp/lbdPf)*np.sinh((L - Lp)/lbdDf)))

    # PSP with unitary current input
    if X<=Lp:
        I = 1./(1+2.*1j*np.pi*f*tauP) # unitary current, input in proximal loc
        if x<=X:
            PSP = B(X,I)*(np.cosh(x/lbdPf)+gPf*np.sinh(x/lbdPf))
        if x>X and x<=Lp:
            PSP = C(X,I)*np.cosh((x-Lp)/lbdPf)+D(X,I)*np.sinh((x-Lp)/lbdPf)
        else: # means x>X and x>Lp
            PSP = A(X,I)*np.cosh((x-L)/lbdDf)
    else:
        print 'case not considered yet !!'
        print 1e6*x, 1e6*X, 1e6*Lp
    if np.trapz(Gf2*np.abs(PSP)**2, f)>1:
        print 1e6*x, 1e6*X
        
    muV_X = stat_pot_function([X], shtn_input, soma, stick, params)[0]

    return np.trapz(Gf2*np.abs(PSP)**2, f)*(Erev-muV_X)**2

def get_the_theoretical_sV_and_Tv(shtn_input, f, x, params, soma, stick,\
                                  precision=50):
    sv2 = np.zeros(len(x))
    Tv = np.zeros(len(x))

    Source_Array = x[1:] # discarding the soma, treated below
    DX = Source_Array[1]-Source_Array[0]
    norm_for_Tv = 0*Tv

    for ix_dest in range(len(x)):

        #### DENDRITIC SYNAPSES
        for ix_source in range(len(Source_Array)): # less intervals than points

            X_source = Source_Array[ix_source]
            if X_source<=stick['L_prox']:
                fe, fi = shtn_input['fe_prox'], shtn_input['fi_prox']
            else:
                fe, fi = shtn_input['fe_dist'], shtn_input['fi_dist']

            # excitatory synapse at dendrites
            Gf2 = exp_FT_mod(f, params['Qe'], params['Te'])
            psp2 = psp_norm_square_integral_per_dend_synapse_type(\
                            x[ix_dest], X_source,\
                            f, Gf2, params['Ee'], shtn_input,\
                            soma, stick, params,
                            precision=precision)
            # psp0 = psp_0_freq_per_dend_synapse_type(\
            #                 x[ix_dest], X_source,\
            #                 params['Qe']*params['Te'], params['Ee'],\
            #                 Garray, soma, stick, params,
            #                 precision=precision)
            sv2[ix_dest] += 2.*np.pi*fe*DX*stick['D']/stick['exc_density']*psp2
            # Tv[ix_dest] += 2.*np.pi*fe*DX*stick['D']/stick['exc_density']*psp0**3/4./psp2
            # norm_for_Tv[ix_dest] += 2.*np.pi*fe*DX*stick['D']/stick['exc_density']*psp0

            # inhibitory synapse at dendrites
            Gf2 = exp_FT_mod(f, params['Qi'], params['Ti'])
            psp2 = psp_norm_square_integral_per_dend_synapse_type(\
                            x[ix_dest], X_source,\
                            f, Gf2, params['Ei'], shtn_input,\
                            soma, stick, params,
                            precision=precision)
            # psp0 = psp_0_freq_per_dend_synapse_type(\
            #                 x[ix_dest], X_source,\
            #                 params['Qi']*params['Ti'], params['Ei'],\
            #                 Garray, soma, stick, params,
            #                 precision=precision)
            sv2[ix_dest] += 2.*np.pi*fi*DX*stick['D']/stick['inh_density']*psp2
            # Tv[ix_dest] += 2.*np.pi*fi*DX*stick['D']/stick['inh_density']*psp0**3/4./psp2
            # norm_for_Tv[ix_dest] += 2.*np.pi*fi*DX*stick['D']/stick['inh_density']*psp0

        # #### SOMATIC SYNAPSES, discret summation, only inhibition
        fi = shtn_input['fi_soma']
        Gf2 = exp_FT_mod(f, params['Qi'], params['Ti'])
        psp2 = psp_norm_square_integral_per_dend_synapse_type(\
                        x[ix_dest], 0.,\
                        f, Gf2, params['Ei'], shtn_input,\
                        soma, stick, params,
                        precision=precision)
        # psp0 = psp_0_freq_per_dend_synapse_type(\
        #                 x[ix_dest], 0.,\
        #                 params['Qi']*params['Ti'], params['Ei'],\
        #                 Garray, soma, stick, params,
        #                 precision=precision)
        sv2[ix_dest] += 2.*np.pi*fi*soma['L']*soma['D']/soma['inh_density']*psp2
        # Tv[ix_dest] += 2.*np.pi*fi*soma['L']*soma['D']/soma['inh_density']*psp0**3/4./psp2
        # norm_for_Tv[ix_dest] += 2.*np.pi*fi*soma['L']*soma['D']/soma['inh_density']*psp0
        
    # return np.sqrt(sv2), Tv/norm_for_Tv
    return np.sqrt(sv2), 0*sv2


def get_the_input_and_transfer_resistance(fe, fi, f, x, params, soma, stick):
    Rin, Rtf = np.zeros(len(x)), np.zeros(len(x))
    Garray = calculate_mean_conductances(fe, fi, soma, stick, params)
    
    for ix_dest in range(len(x)):

        Rtf[ix_dest] = dv_kernel_per_I(0., x[ix_dest],
                0., Garray, stick, soma, params) # at 0 frequency
        Rin[ix_dest] = dv_kernel_per_I(x[ix_dest], x[ix_dest],
                0., Garray, stick, soma, params) # at 0 frequency
    return Rin, Rtf

