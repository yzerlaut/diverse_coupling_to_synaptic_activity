import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit, leastsq
import scipy.special as sp_spec
import statsmodels.api as sm

## NORMALIZING COEFFICIENTS
# needs to be global here, because used both in the function
# and its derivatives
muV0, DmuV0 = -60e-3,10e-3
sV0, DsV0 =4e-3, 6e-3
TvN0, DTvN0 = 0.5, 1.

### CORRECTIONS FOR THE EFFECTIVE THRESHOLD

def final_threshold_func(coeff, muV, sV, TvN, muGn):
        
    output = coeff[0]+\
             coeff[1]*(muV-muV0)/DmuV0+\
             coeff[2]*(sV-sV0)/DsV0+\
             coeff[3]*(TvN-TvN0)/DTvN0

    return output

### FUNCTION, INVERSE FUNCTION

def erfc_func(mu, sigma, TvN, Vthre, Gl, Cm):
    return .5/TvN*Gl/Cm*\
      sp_spec.erfc((Vthre-mu)/np.sqrt(2)/sigma)

def effective_Vthre(Y, mVm, sVm, TvN, Gl, Cm):
    Vthre_eff = mVm+np.sqrt(2)*sVm*sp_spec.erfcinv(\
                    Y*2.*TvN*Cm/Gl) # effective threshold
    return Vthre_eff

def final_func(coeff, muV, sV, TvN, Gl, Cm):
    return erfc_func(muV, sV, TvN, final_threshold_func(coeff, muV, sV, TvN, 1.), Gl, Cm)

## all derivatives

def derivatives_template(P, muV, sV, TvN, muGn, El, Gl, Cm,\
                         DmuV0=DmuV0, DsV0=DsV0, DTvN0=DTvN0):
    # in common
    Vthre = final_threshold_func(P, muV, sV, TvN, muGn)
    exp_term = np.exp(-((Vthre-muV)/np.sqrt(2)/sV)**2)
    denom = np.sqrt(2.*np.pi)*TvN*Cm/Gl*sV
    Fout = erfc_func(muV, sV, TvN, Vthre, Gl, Cm)
    # independent
    factor1 = (1-P[1]/DmuV0)
    factor2 = ((Vthre-muV)/sV-P[2]/DsV0)
    factor3 = P[3]/DTvN0 # TvN needs additional Fout term !!
    return factor1*exp_term/denom, factor2*exp_term/denom,\
        -(factor3*exp_term/denom+Fout/TvN)


def linear_fitting_of_threshold_with_firing_weight(\
            Fout, muV, sV, TvN, muGn, Gl, Cm, El,\
            maxiter=1e5, xtol=1e-18,
            print_things=True):

    # we start by comuting the threshold
    i_non_zeros = np.nonzero(Fout)
    muV2, sV2, TvN2, muGn2, Fout2 = \
      muV[i_non_zeros], sV[i_non_zeros],\
       TvN[i_non_zeros], muGn[i_non_zeros],\
       Fout[i_non_zeros]
    
    vthre = effective_Vthre(Fout2, muV2, sV2, TvN2, Gl, Cm)

    P = np.zeros(4)
    P[0] = -45e-3 # just threshold in the right range

    def Res(p):
        threshold2 = final_threshold_func(p, muV2, sV2, TvN2, muGn2)
        to_minimize = (vthre-threshold2)**2
        return np.mean(to_minimize)/len(threshold2)

    # bnds = ((-90e-3, -10e-3), (None,None), (None,None), (None,None), (None,None),\
    #         (None,None),(-2e-3, 3e-3), (-2e-3, 3e-3))
    # plsq = minimize(Res,P, method='SLSQP', bounds=bnds, tol=xtol,\
    #         options={'maxiter':maxiter})

    plsq = minimize(Res,P, tol=xtol, options={'maxiter':maxiter})
            
    P = plsq.x
    if print_things:
        print plsq
    return P

def fitting_Vthre_then_Fout(Fout, muV, sV, TvN, muGn, Gl, Cm, El,\
        maxiter=1e5, ftol=1e-15,\
        print_things=True,\
        return_chi2=False):

    P = linear_fitting_of_threshold_with_firing_weight(\
            Fout, muV, sV, TvN, muGn, Gl, Cm, El,\
            maxiter=maxiter, xtol=ftol,\
            print_things=print_things)

    def Res(p, muV, sV, TvN, muGn, Fout):
        return (Fout-erfc_func(muV, sV, TvN,\
           final_threshold_func(p, muV, sV, TvN, muGn), Gl, Cm))
                                
    if return_chi2:
        P,cov,infodict,mesg,ier = leastsq(
            Res,P, args=(muV, sV, TvN, muGn, Fout),\
            full_output=True)
        ss_err=(infodict['fvec']**2).sum()
        ss_tot=((Fout-Fout.mean())**2).sum()
        rsquared=1-(ss_err/ss_tot)
        return P, rsquared
    else:
        P = leastsq(Res, P, args=(muV, sV, TvN, muGn, Fout))[0]
        return P
        
        
