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
