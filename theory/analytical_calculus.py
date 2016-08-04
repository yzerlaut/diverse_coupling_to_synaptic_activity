# from numba import jit
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import sys
sys.path.append('../code/')
import fourier_for_real as rfft

def params_for_cable_theory(cable, params):
    """
    applies the radial symmetry, to get the linear cable theory parameters
    """
    D = cable['D']
    cable['rm'] = 1./params['g_pas']/(np.pi*D) # [O.m]   """" NEURON 1e-1 !!!! """" 
    cable['ri'] = params['Ra']/(np.pi*D**2/4) # [O/m]
    cable['cm'] = params['cm']*np.pi*D # [F/m] """" NEURON 1e-2 !!!! """"
    

def setup_model(soma, stick, params, verbose=True):
    """ returns the different diameters of the equivalent cylinder
    given a number of branches point"""
    params['EqCylinder'] = np.linspace(0, 1, stick['B']+1)*stick['L'] # equally space branches !
    cables, xtot = [], np.zeros(1)
    cables.append(soma.copy())
    cables[0]['inh_density'] = soma['inh_density']
    cables[0]['exc_density'] = soma['exc_density']
    Ke_tot, Ki_tot = 0, 0
    D = stick['D'] # mothers branch diameter
    for i in range(1,len(params['EqCylinder'])):
        cable = stick.copy()
        cable['x1'], cable['x2'] = params['EqCylinder'][i-1], params['EqCylinder'][i]
        cable['L'] = cable['x2']-cable['x1']
        x = np.linspace(cable['x1'], cable['x2'], cable['NSEG']+1)
        cable['x'] = .5*(x[1:]+x[:-1])
        xtot = np.concatenate([xtot, cable['x']])
        cable['D'] = D*2**(-2*(i-1)/3.)
        cable['inh_density'] = stick['inh_density']
        cable['exc_density'] = stick['exc_density']
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
        cable['Area_per_seg'] = cable['L']*cable['D']*np.pi/cable['NSEG']
        if cable['name']!='soma':
            jj+=1
    if verbose:
        print "Total number of EXCITATORY synapses : ", Ke_tot
        print "Total number of INHIBITORY synapses : ", Ki_tot
    # we store this info in the somatic comp
    cables[0]['Ke_tot'], cables[0]['Ki_tot'] = Ke_tot, Ki_tot
    return xtot, cables

def calculate_mean_conductances(shtn_input,\
                                soma, cable, params):
    """
    this calculates the conductances that needs to plugged in into the
    linear cable equation
    Note :
    the two first conductances are RADIAL conductances !! (excitation and
    inhibition along the cable)
    the third one is an ABSOLUTE conductance !
    """

    D = cable['D']
        
    Te_prox, Qe_prox = params['Te'], params['Qe']
    Ti_prox, Qi_prox = params['Ti'], params['Qi']
    Qe_dist = Qe_prox*params['factor_for_distal_synapses_weight']
    Qi_dist = Qi_prox*params['factor_for_distal_synapses_weight']
    Te_dist = Te_prox*params['factor_for_distal_synapses_tau']
    Ti_dist = Ti_prox*params['factor_for_distal_synapses_tau']
        
    ge_prox = Qe_prox*shtn_input['fe_prox']*Te_prox*D*np.pi/cable['exc_density']
    gi_prox = Qi_prox*shtn_input['fi_prox']*Ti_prox*D*np.pi/cable['inh_density']
    ge_dist = Qe_dist*shtn_input['fe_dist']*Te_dist*D*np.pi/cable['exc_density']
    gi_dist = Qi_dist*shtn_input['fi_dist']*Ti_dist*D*np.pi/cable['inh_density']

    # somatic inhibitory conductance
    Ls, Ds = soma['L'], soma['D']
    Kis = np.pi*Ls*Ds/soma['inh_density']
    Gi_soma = Qi_prox*Kis*shtn_input['fi_prox']*Ti_prox
    
    return Gi_soma, ge_prox, gi_prox, ge_dist, gi_dist


def ball_and_stick_params(soma, stick, params):
    Ls, Ds = soma['L'], soma['D']
    L, D = stick['L'], stick['D']
    Lp = L*params['fraction_for_L_prox']
    Rm = 1./(np.pi*Ls*Ds*params['g_pas'])
    # print 'Rm (soma)', 1e-6*Rm, 'MOhm'
    Cm = np.pi*Ls*Ds*params['cm']
    # print 'Cm (soma)', 1e-12*Cm, 'pF'
    El, Ei, Ee = params['El'], params['Ei'], params['Ee']
    rm, cm, ri = stick['rm'], stick['cm'], stick['ri']
    return Ls, Ds, L, D, Lp, Rm, Cm, El, Ee, Ei, rm, cm, ri


def ball_and_stick_constants(shtn_input, soma, stick, params):
    Ls, Ds, L, D, Lp, Rm, Cm, El, Ee, Ei, rm, cm, ri = \
                    ball_and_stick_params(soma, stick, params)
    Gi_soma, ge_prox, gi_prox, ge_dist, gi_dist = \
                calculate_mean_conductances(shtn_input,\
                                            soma, stick, params)
    tauP = rm*cm/(1+rm*ge_prox+rm*gi_prox)
    lbdP = np.sqrt(rm/ri/(1+rm*ge_prox+rm*gi_prox))
    tauD = rm*cm/(1+rm*ge_dist+rm*gi_dist)
    lbdD = np.sqrt(rm/ri/(1+rm*ge_dist+rm*gi_dist))
    tauS = Rm*Cm/(1+Rm*Gi_soma)
    return tauS, tauP, lbdP, tauD, lbdD

def cable_eq_params(f, tauS, tauP, lbdP, tauD, lbdD, Cm, cm, ri, lp, l, B):
        # proximal params
    afP = np.sqrt(1+2.*1j*np.pi*f*tauP)
    gfP = lbdP*Cm*ri/tauS*(1+2.*1j*np.pi*f*tauS)
    rfP = tauP/cm/lbdP

    # distal params
    afD = np.sqrt(1+2.*1j*np.pi*f*tauD)
    rfD = tauD/cm/lbdD

    # ball and tree rescaling
    Lp = rescale_x(lp, l, lp, B, lbdP, lbdD)
    L = rescale_x(l, l, lp, B, lbdP, lbdD)
    
    return afP, gfP, rfP, afD, rfD, Lp, L


############### INPUT FROM SYMPY ###################

exec(open('../theory/functions.txt'))

def lbd(x, l, lp, B, lbdP, lbdD):
    # specific to evenly space branches !! (see older implementation with EqCylinder for more general implement.)
    branch_length = l/B # length of one branch !
    reduction_factor =  np.power(2., -1./3.*np.intp(x/branch_length))
    new_lbd = (lbdP+(lbdD-lbdP)*.5*(np.sign(x+1e-9-lp)+1))
    return new_lbd*reduction_factor
    
def rescale_x(x, l, lp, B, lbdP, lbdD):
    # specific to evenly space branches !! (see older implementation with EqCylinder for more general implement.)
    EqCylinder = np.linspace(0, l, B+1)
    C = EqCylinder[EqCylinder<=x]
    return np.sum(np.diff(C)/lbd(C, l, lp, B, lbdP, lbdD)[:-1])+(x-C[-1])/lbd(C[-1], l, lp, B, lbdP, lbdD)

def stat_pot_function(x, shtn_input, soma, stick, params):

    params_for_cable_theory(stick, params)

    Ls, Ds, l, D, lp, Rm, Cm,\
        El, Ee, Ei, rm, cm, ri = ball_and_stick_params(soma, stick, params)

    Gi_soma, ge_prox, gi_prox, ge_dist, gi_dist = \
                calculate_mean_conductances(shtn_input,\
                                            soma, stick, params)
    tauS, tauP, lbdP, tauD, lbdD = \
            ball_and_stick_constants(shtn_input, soma, stick, params)

    # proximal params
    v0P = (El+rm*ge_prox*Ee+rm*gi_prox*Ei)/(1+rm*ge_prox+rm*gi_prox)
    gP = Cm*ri*lbdP/tauS
    # distal params
    v0D = (El+rm*ge_dist*Ee+rm*gi_dist*Ei)/(1+rm*ge_dist+rm*gi_dist)
    # somatic params
    V0 = (El+Rm*Gi_soma*Ei)/(1+Rm*Gi_soma)

    X = np.array([rescale_x(xx, l, lp, stick['B'], lbdP, lbdD) for xx in x])
    Lp, L = rescale_x(lp, l, lp, stick['B'], lbdP, lbdD), rescale_x(l, l, lp, stick['B'], lbdP, lbdD)

    return np.array([muV_prox(XX, Lp, L, lbdD, lbdP, gP, v0D, v0P, V0) if XX<Lp\
                     else muV_dist(XX, Lp, L, lbdD, lbdP, gP, v0D, v0P, V0) for XX in X])

# @jit
def exp_FT(f, Q, Tsyn, t0=0):
    return Q*np.exp(-1j*2*np.pi*t0*f)/(1j*2*np.pi*f+1./Tsyn)

# @jit
def exp_FT_mod(f, Q, Tsyn):
    return Q**2/((2*np.pi*f)**2+(1./Tsyn)**2)

# @jit
def split_root_square_of_imaginary(f, tau):
    # returns the ral and imaginary part of Sqrt(1+2*Pi*f*tau)
    af = np.sqrt((np.sqrt(1+(2*np.pi*f*tau)**2)+1)/2)
    bf = np.sqrt((np.sqrt(1+(2*np.pi*f*tau)**2)-1)/2)
    return af, bf

# @jit
def psp_norm_square_integral_per_dend_synapse_type(x_dest, x_src, f, Gf2,\
                            Erev, shtn_input,\
                            soma, stick, params,
                            precision=1e2):

    # muV for mean driving force
    muV_X = stat_pot_function([x_src], shtn_input, soma, stick, params)[0]
    
    # model parameters
    Ls, Ds, l, D, lp, Rm, Cm,\
        El, Ee, Ei, rm, cm, ri = ball_and_stick_params(soma, stick, params)
    # activity dependent parameters
    tauS, tauP, lbdP, tauD, lbdD = \
            ball_and_stick_constants(shtn_input, soma, stick, params)
    # cable constants
    afP, gfP, rfP, afD, rfD, Lp, L = cable_eq_params(f, tauS, tauP, lbdP,\
                                    tauD, lbdD, Cm, cm, ri, lp, l, stick['B'])
    Xsrc = rescale_x(x_src, l, lp, stick['B'], lbdP, lbdD)
    Xdest = rescale_x(x_dest, l, lp, stick['B'], lbdP, lbdD)

    # PSP with unitary current input
    if Xsrc<=Lp:
        if Xdest<=Xsrc:
            PSP = dv_X_Xsrc_Lp(Xdest, Xsrc, 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)
        elif Xdest>Xsrc and Xdest<=Lp:
            PSP = dv_Xsrc_X_Lp(Xdest, Xsrc, 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)
        elif Xdest>Xsrc and Xdest>Lp:
            PSP = dv_Xsrc_Lp_X(Xdest, Xsrc, 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)
    elif Xsrc>Lp:
        if Xdest<=Lp:
            PSP = dv_X_Lp_Xsrc(Xdest, Xsrc, 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)
        elif Xdest>Lp and Xdest<=Xsrc:
            PSP = dv_Lp_X_Xsrc(Xdest, Xsrc, 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)
        elif Xdest>Xsrc:
            PSP = dv_Lp_Xsrc_X(Xdest, Xsrc, 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)
    return Gf2*np.abs(PSP)**2*(Erev-muV_X)**2

# # @jit
def get_the_theoretical_sV_and_Tv(shtn_input,\
                                  f, x, params, soma, stick,\
                                  precision=50):
    Pv = np.zeros((len(x), len(f))) # power spectral density of Vm for each position

    Source_Array = np.linspace(x.min(), x.max(), precision+1)
    Source_Array = .5*(Source_Array[1:]+Source_Array[:-1]) # discarding the soma, treated below
    DX = Source_Array[1]-Source_Array[0]
    Branch_weights = 0*Source_Array # initialized t0 0 !

    synchrony = shtn_input['synchrony']
    synch_factor = (1-synchrony)+2**2*(synchrony-synchrony**2)+\
      3**2*(synchrony**2-synchrony**3)+4**2*synchrony**3
    synch_dividor= 1+synchrony+synchrony**2+synchrony**3
    # synch_factor = 1.+3.*synchrony # factor for the PSP event
    # synch_dividor = 1.+synchrony # dividor for the real frequency
    # rational: new_freq = freq/(1.+synchrony)
    # we generate spikes with frequency : new_freq
    # then we duplicate each spike with a probability : 'synchrony'
    # so we get independent events (1 spike) with frequency new_freq*(1-synchrony)
    # and we get double events with freq: new_freq*synchrony
    # double envents have a weight (2*PSP)**2 -> 4
    # new_freq*(1-synchrony+4*synchrony)

    
    # for b in params['EqCylinder']:
    #     Branch_weights[np.where(Source_Array>=b)[0]] += 1
    # Branch_weights = np.power(2., Branch_weights-1) # NUMBER OF BRANCHES

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
            Qe, Qi = weight_synapse_factor*params['Qe'],\
                     weight_synapse_factor*params['Qi']

            # excitatory synapse at dendrites
            Gf2 = exp_FT_mod(f, Qe, params['Te']*tau_synapse_factor)
            psp2 = psp_norm_square_integral_per_dend_synapse_type(\
                            x[ix_dest], X_source,\
                            f, Gf2, params['Ee'], shtn_input,\
                            soma, stick, params, precision=precision)
            Pv[ix_dest,:] += np.pi*fe*DX*stick['D']/stick['exc_density']*psp2*synch_factor

            # inhibitory synapse at dendrites
            Gf2 = exp_FT_mod(f, Qi, params['Ti']*tau_synapse_factor)
            psp2 = psp_norm_square_integral_per_dend_synapse_type(\
                            x[ix_dest], X_source,\
                            f, Gf2, params['Ei'], shtn_input,\
                            soma, stick, params, precision=precision)
            Pv[ix_dest,:] += np.pi*fi*DX*stick['D']/stick['inh_density']*psp2*synch_factor

        # #### SOMATIC SYNAPSES, discret summation, only inhibition, no branch weighting
        fi = shtn_input['fi_prox']
        fi /= synch_dividor
        Gf2 = exp_FT_mod(f, params['Qi'], params['Ti'])
        psp2 = psp_norm_square_integral_per_dend_synapse_type(\
                        x[ix_dest], 0.,\
                        f, Gf2, params['Ei'], shtn_input,\
                        soma, stick, params, precision=precision)
        Pv[ix_dest,:] += np.pi*fi*soma['L']*soma['D']/soma['inh_density']*psp2*synch_factor

    sV2, Tv = np.zeros(len(x)), np.zeros(len(x))
    for ix in range(len(x)):
        sV2[ix] = 2.*np.trapz(np.abs(Pv[ix,:]), f)
        Tv[ix] = .5*Pv[ix,0]/(2.*np.trapz(np.abs(Pv[ix,:]), f)) # 2 times the integral to have from -infty to +infty (and methods gives [0,+infty])
        
    return np.sqrt(sV2), Tv

# @jit
def psp_norm_square_integral_per_dend_synapse_type_at_soma(x_src, f, Gf2,\
                            Erev, shtn_input,\
                            soma, stick, params,
                            precision=1e2):

    # muV for mean driving force
    muV_X = stat_pot_function([x_src], shtn_input,\
                              soma, stick, params)[0]
    
    # model parameters
    Ls, Ds, l, D, lp, Rm, Cm,\
        El, Ee, Ei, rm, cm, ri = ball_and_stick_params(soma, stick, params)
    # activity dependent parameters
    tauS, tauP, lbdP, tauD, lbdD = \
            ball_and_stick_constants(shtn_input, soma, stick, params)
    # cable constants
    afP, gfP, rfP, afD, rfD, Lp, L = cable_eq_params(f, tauS, tauP, lbdP,\
                                    tauD, lbdD, Cm, cm, ri, lp, l, stick['B'])
    Xsrc = rescale_x(x_src, l, lp, stick['B'], lbdP, lbdD)
    
    # PSP with unitary current input
    if Xsrc<=Lp:
        # to be replaced by simplified expression
        PSP = dv_X_Xsrc_Lp(0., Xsrc, 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)
    elif Xsrc>Lp:
        # to be replaced by simplified expression
        PSP = dv_X_Lp_Xsrc(0., Xsrc, 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)

    return Gf2*np.abs(PSP)**2*(Erev-muV_X)**2

# @jit
def get_the_fluct_prop_at_soma(SHTN_INPUT, params, soma, stick,\
                               precision=100, f=rfft.time_to_freq(1000, 1e-4)):

    params_for_cable_theory(stick, params)
    setup_model(soma, stick, params)


    # check if the shtn input is an array
    n = len(SHTN_INPUT['fi_prox'])

    muV, sV, TvN, muGn = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        
    # input resistance at rest
    Rin0 = get_the_input_resistance_at_soma(soma, stick, params,
                                            {'fi_prox':0, 'fe_prox':0,'fi_prox':0,
                                             'fe_dist':0,'fi_dist':0, 'synchrony':0.})
    # membrane time constant at rest
    Tm0 = get_membrane_time_constants(soma, stick, params)

    # then temporal loop
    for i in range(n):
        
        shtn_input = {'fi_prox':SHTN_INPUT['fi_prox'][i], 'fe_prox':SHTN_INPUT['fe_prox'][i],\
                      'fi_prox':SHTN_INPUT['fi_prox'][i], 'fe_dist':SHTN_INPUT['fe_dist'][i],\
                      'fi_dist':SHTN_INPUT['fi_dist'][i], 'synchrony':SHTN_INPUT['synchrony'][i]}

        synchrony = shtn_input['synchrony'] # Note 4 events maximum !!!
        synch_factor = (1-synchrony)+2**2*(synchrony-synchrony**2)+\
          3**2*(synchrony**2-synchrony**3)+4**2*synchrony**3
        synch_dividor= 1+synchrony+synchrony**2+synchrony**3

        Pv = np.zeros(len(f)) # power spectral density of Vm for each position

        Source_Array = np.linspace(0, stick['L'], precision+1)
        Source_Array = .5*(Source_Array[1:]+Source_Array[:-1]) # discarding the soma, treated below
        DX = Source_Array[1]-Source_Array[0]
        # Branch_weights = 0*Source_Array # initialized t0 0 !
        # for b in params['EqCylinder']:
        #     Branch_weights[np.where(Source_Array>=b)[0]] += 1
        # Branch_weights = np.power(2., Branch_weights-1) # NUMBER OF BRANCHES

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
                tau_synapse_factor = 1. # removed params

            ## weighting due to branching !
            # fe, fi = fe*Branch_weights[ix_source], fi*Branch_weights[ix_source]
            fe /= synch_dividor
            fi /= synch_dividor
            Qe, Qi = weight_synapse_factor*params['Qe'],\
                     weight_synapse_factor*params['Qi']
            # Qe, Qi = weight_synapse_factor*params['Qe']/Branch_weights[ix_source],\
            #          weight_synapse_factor*params['Qi']/Branch_weights[ix_source]

            # excitatory synapse at dendrites
            Gf2 = exp_FT_mod(f, Qe, params['Te']*tau_synapse_factor)
            psp2 = psp_norm_square_integral_per_dend_synapse_type_at_soma(\
                            X_source,\
                            f, Gf2, params['Ee'], shtn_input,\
                            soma, stick, params, precision=precision)
            Pv += np.pi*fe*DX*stick['D']/stick['exc_density']*psp2*synch_factor

            # inhibitory synapse at dendrites
            Gf2 = exp_FT_mod(f, Qi, params['Ti']*tau_synapse_factor)
            psp2 = psp_norm_square_integral_per_dend_synapse_type_at_soma(\
                            X_source,\
                            f, Gf2, params['Ei'], shtn_input,\
                            soma, stick, params, precision=precision)
            Pv += np.pi*fi*DX*stick['D']/stick['inh_density']*psp2*synch_factor

        # #### SOMATIC SYNAPSES, discret summation, only inhibition, no branch weighting
        fi = shtn_input['fi_prox']
        Gf2 = exp_FT_mod(f, params['Qi'], params['Ti'])
        psp2 = psp_norm_square_integral_per_dend_synapse_type_at_soma(\
                        0.,\
                        f, Gf2, params['Ei'], shtn_input,\
                        soma, stick, params, precision=precision)
        Pv += np.pi*fi*soma['L']*soma['D']/soma['inh_density']*psp2*synch_factor

        Rin = get_the_input_resistance_at_soma(soma, stick, params, shtn_input)

        muV[i] = stat_pot_function([0], shtn_input, soma, stick, params)[0]

        sV[i] = np.sqrt(2.*np.trapz(np.abs(Pv), f))

        TvN[i] = .5*Pv[0]/(2.*np.trapz(np.abs(Pv), f))/Tm0 # 2 times the integral to have from -infty to +infty (and methods gives [0,+infty])

        muGn[i] = Rin0/Rin
        
    return muV, sV, TvN, muGn


def get_the_input_resistance_at_soma(soma, stick, params, shtn_input):

    # model parameters
    Ls, Ds, l, D, lp, Rm, Cm,\
        El, Ee, Ei, rm, cm, ri = ball_and_stick_params(soma, stick, params)
    # activity dependent parameters
    tauS, tauP, lbdP, tauD, lbdD = \
            ball_and_stick_constants(shtn_input, soma, stick, params)
    # cable constants
    afP, gfP, rfP, afD, rfD, Lp, L = cable_eq_params(0., tauS, tauP, lbdP,\
                                    tauD, lbdD, Cm, cm, ri, lp, l, stick['B'])
                                    
    # PSP with unitary current input
    # input and recording in x=0 
    return np.abs(dv_X_Xsrc_Lp(0., 0., 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP))

def get_the_transfer_resistance_to_soma(soma, stick, params, precision=100):

    shtn_input = {'fi_prox':0, 'fe_prox':0,'fi_prox':0,
                  'fe_dist':0,'fi_dist':0}
    params_for_cable_theory(stick, params)
    setup_model(soma, stick, params)
    
    # model parameters
    Ls, Ds, l, D, lp, Rm, Cm,\
        El, Ee, Ei, rm, cm, ri = ball_and_stick_params(soma, stick, params)
    # activity dependent parameters
    tauS, tauP, lbdP, tauD, lbdD = \
            ball_and_stick_constants(shtn_input, soma, stick, params)
    # cable constants
    afP, gfP, rfP, afD, rfD, Lp, L = cable_eq_params(0., tauS, tauP, lbdP,\
                                    tauD, lbdD, Cm, cm, ri, lp, l, stick['B'])
                                    
    Source_Array = np.linspace(0, stick['L'], precision+1)
    Source_Array = .5*(Source_Array[1:]+Source_Array[:-1]) # discarding the soma, treated below
    DX = Source_Array[1]-Source_Array[0]
    Branch_weights = 0*Source_Array # initialized t0 0 !
    for b in params['EqCylinder']:
        Branch_weights[np.where(Source_Array>=b)[0]] += 1
        
    Branch_weights = np.power(2., Branch_weights-1) # NUMBER OF BRANCHES

    R_transfer = np.zeros(len(Source_Array)+1)
    N_synapses = np.zeros(len(Source_Array)+1)
    
    #### DENDRITIC SYNAPSES
    for ix_source in range(len(Source_Array)): 

        X_source = Source_Array[ix_source]
        Xsrc = rescale_x(X_source, l, lp, stick['B'], lbdP, lbdD) # rescaled x_source
        if X_source<=Lp:
            R_transfer[ix_source] = dv_X_Xsrc_Lp(0., Xsrc, 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)
        else:
            R_transfer[ix_source] = dv_X_Lp_Xsrc(0., Xsrc, 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)
        N_synapses[ix_source] = Branch_weights[ix_source]*np.pi*DX*stick['D']*(1./stick['exc_density']+1./stick['inh_density'])

    ### SOMATIC SYNAPSES
    N_synapses[-1] = np.pi*soma['L']*soma['D']*(1./soma['exc_density']+1./soma['inh_density'])
    R_transfer[-1] = dv_X_Xsrc_Lp(0., 0., 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)
    
    return R_transfer, N_synapses

def get_the_mean_transfer_resistance_to_soma(soma, stick, params, precision=100):
    R_transfer, N_synapses = get_the_transfer_resistance_to_soma(soma, stick, params,precision=precision)
    return np.sum(R_transfer*(N_synapses/N_synapses.sum()))

def get_the_input_impedance_at_soma(f, soma, stick, params):

    # model parameters
    Ls, Ds, l, D, lp, Rm, Cm,\
        El, Ee, Ei, rm, cm, ri = ball_and_stick_params(soma, stick, params)
        
    shtn_input = {'fi_prox':0, 'fe_prox':0,'fi_prox':0,
                  'fe_dist':0,'fi_dist':0}
    
    # activity dependent parameters
    tauS, tauP, lbdP, tauD, lbdD = \
            ball_and_stick_constants(shtn_input, soma, stick, params)
            
    # cable constants
    afP, gfP, rfP, afD, rfD, Lp, L = cable_eq_params(f, tauS, tauP, lbdP,\
                                    tauD, lbdD, Cm, cm, ri, lp, l, stick['B'])
                                    
    # PSP with unitary current input
    # input and recording in x=0 
    return dv_X_Xsrc_Lp(0., 0., 1., afP, afD, gfP, rfP, rfD, Lp, L, lbdD, lbdP)

def get_membrane_time_constants(soma, stick, params,
                                f=rfft.time_to_freq(1000, 1e-4)):
    psd = np.abs(get_the_input_impedance_at_soma(f, soma, stick, params))**2
    return .5*psd[0]/(2.*np.trapz(np.abs(psd), f))


def find_balance_at_soma(Fe_prox, Fe_dist, fe0, params, soma, stick,\
                         balance=-60e-3, precision=1e2):

    # fe0 is a baseline excitation shared between prox and distal
    
    shtn_input = {'fe_prox':fe0+Fe_prox, 'fe_dist':fe0+Fe_dist}
    FI_prox = np.linspace(0., 10.*Fe_prox, int(precision))
    FI_dist = np.linspace(0., 10.*Fe_dist, int(precision))

    muV = 0.*FI_dist
    for i in range(int(precision)):
        shtn_input['fi_prox'], shtn_input['fi_dist']= FI_prox[i], FI_dist[i]
        muV[i] = stat_pot_function([0], shtn_input, soma, stick, params)[0]
    i0 = np.argmin(np.abs(muV-balance))
    return FI_prox[i0], FI_dist[i0]

def find_baseline_excitation(params, soma, stick,\
                             f_min=0, f_max=3.,
                             balance=-60e-3, synch=0.5,
                             precision=1e2):

    f = np.linspace(f_min, f_max, precision)
    shtn_input = {'fi_prox':0., 'fi_dist': 0, 'synch':synch}
    muV = 0.*f
    for i in range(int(precision)):
        shtn_input['fe_prox'], shtn_input['fe_dist']= f[i], f[i]
        muV[i] = stat_pot_function([0], shtn_input, soma, stick, params)[0]
    i0 = np.argmin(np.abs(muV-balance))
    return f[i0]
