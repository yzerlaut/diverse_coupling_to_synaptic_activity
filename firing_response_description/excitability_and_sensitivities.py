import numpy as np

from template_and_fitting import erfc_func, final_threshold_func,\
     derivatives_template

low_rate_fluct_domain  = {
'Ts_ratio':.15,
'muG_min':1.5,
'muG_max':8.,
'sV_min':2e-3,
'sV_max':7e-3,
'muV_min':-90e-3,
'muV_max':-10e-3,
'max_Fout':15.,
'min_Fout':1.}


def find_excitability_region(P, El, Gl, Cm,\
                             pp = low_rate_fluct_domain,
                             discret=40,\
                             print_domain=False):
    """
    P are the coefficients of the firing rate response fit
    the other parameters set the default boundaries of
    the numerical evaluation of the relevant firing rate domain

    here all in SI units
    """
    sV0 = np.linspace(pp['sV_min'], pp['sV_max'], discret)
    TvN0 = np.linspace(pp['Ts_ratio']+1./pp['muG_max'], pp['Ts_ratio']+1./pp['muG_min'], discret)
    muV0 = np.linspace(pp['muV_min'], pp['muV_max'], 5*discret) # high discret
    muV, sV, TvN = np.meshgrid(muV0, sV0, TvN0) 

    Vthre = final_threshold_func(P, muV, sV, TvN, Gl)
    Fout = erfc_func(muV, sV, TvN, Vthre, Gl, Cm)

    dmuV, dsV, dTvN = derivatives_template(\
            P, muV, sV, TvN, Gl*np.ones(muV.shape), El, Gl, Cm)
            
    ok = (Fout<pp['max_Fout']) & (Fout>pp['min_Fout']) & (dmuV>0) & (dsV>0) # desired conditions

    if print_domain:
        print 'muV domain :', round(1e3*muV[ok].min()), round(1e3*muV[ok].max())
        print 'sV domain :', round(1e3*sV[ok].min()), round(1e3*sV[ok].max())
        print 'TvN domain :', round(1e2*TvN[ok].min()), round(1e2*TvN[ok].max())
    
    return Fout[ok], Vthre[ok], muV[ok], sV[ok], TvN[ok]

def get_mean_encoding_power(P, El, Gl, Cm,\
                            pp = low_rate_fluct_domain,
                            discret=20, with_rescaling=False):
    """
    returns the mean excitability as well as the mean encoding coeffs in:
    mV, Hz/mV, Hz/mV and Hz/%
    """
    Fout, Vthre, muV, sV, TvN = find_excitability_region(P, El, Gl, Cm,\
                                    pp=pp, discret=discret)
    dmuV, dsV, dTvN = derivatives_template(\
            P, muV, sV, TvN, Gl*np.ones(muV.shape), El, Gl, Cm)

    return 1e3*Vthre.mean(), dmuV.mean()*1e-3, dsV.mean()*1e-3, 1e-2*dTvN.mean()



if __name__=='__main__':
    P = [-45e-3,0,0,0]
    El, Gl, Cm = -70e-3,10e-9,200e-12
    Fout, Vthre, muV, sV, TvN = find_excitability_region(P, El, Gl, Cm)
    print derivative_sV(P, muV, sV, TvN, 0*muV, El, Gl, Cm).mean()*1e-3


