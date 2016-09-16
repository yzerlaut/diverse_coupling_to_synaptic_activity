#!/usr/bin/python
# Filename: signanalysis.py

import numpy.fft as fft
import numpy as np
from scipy import signal
from scipy import integrate

def autocorrel(signal, tmax, dt):
    """
    argument : signal (np.array), tmax and dt (float)
    tmax, is the maximum length of the autocorrelation that we want to see
    returns : autocorrel (np.array), time_shift (np.array)
    take a signal of time sampling dt, and returns its autocorrelation
     function between [0,tstop] (normalized) !!
    """
    steps = int(tmax/dt) # number of steps to sum on
    signal = (signal-signal.mean())/signal.std()
    cr = np.correlate(signal[steps:],signal)/steps
    time_shift = np.arange(len(cr))*dt
    return cr/cr.max(),time_shift


def crosscorrel(signal1,signal2):
    """
    argument : signal1 (np.array()), signal2 (np.array())
    returns : np.array()
    take two signals, and returns their crosscorrelation function 
    """
    signal1 = (signal1-signal1.mean())
    signal2 = (signal2-signal2.mean())
    cr = np.correlate(signal1,signal2,"full")/signal1.std()/signal2.std()
    return cr

def crosscorrel_norm(signal1,signal2):
    """
    computes the cross-correlation, and takes into account the boundary
    effects ! so normalizes by the weight of the bins !!
    the two array have t have the same size @
    
    argument : signal1 (np.array()), signal2 (np.array())
    returns : np.array()
    take two signals, and returns their crosscorrelation function 
    """
    if signal1.size!=signal2.size:
        print("problem no equal size vectors !!")
    signal1 = (signal1-signal1.mean())
    signal2 = (signal2-signal2.mean())
    cr = signal.fftconvolve(signal1,signal2,"full")/signal1.std()/signal2.std()
    ww = np.linspace(signal1.size,0,-1)
    bin_weight = np.concatenate((ww[::-1],ww[1:]))
    return cr/bin_weight



def lowpass(Signal0,dt,cutoff_freq,N_order=int(1e3)):
    """
    argument : Signal0 (np.array()),
    dt : acquisistion time step (real) in SECOND
    cutoff_frequency (real) in HERZ
    N_order : order of the filter !! by default : 1000
    
    returns : np.array() signal filtered at cuttoff freq

    USES : 
    ** signal.firwin(N, cutoff, width=None, window='hamming')
    FIR Filter Design using windowed ideal filter method.
    ** scipy.signal.lfilter(b, a, x, axis=-1, zi=None)
    Filter data along one-dimension with an IIR or FIR filter.
    """
    # N.B. in firwin, requencies are normalized to the Nyquist frequency, 
    # which is half the sampling rate !!
    freq_acq = 1.0/dt 
    freq_cutoff = cutoff_freq/(freq_acq/2) # cuttoff renormalized

    a=1 # filter denominator, in those cases simple convlution
    # -- On construit le filtre lineaire low pass !!
    b = signal.firwin(N_order,cutoff=freq_cutoff, #,width=freq_cutoff,\
                      window='hamming')
    #filter numerator, here is the low pass 
    return signal.lfilter(b,a,Signal0), b 

def highpass(Signal0,dt,cutoff_freq,N_order=int(1e3)):
    """
    argument : Signal0 (np.array()),
    dt : acquisistion time step (real) in SECOND
    cutoff_frequency (real) in HERZ
    N_order : order of the filter !! by default : 1000
    
    returns : np.array() signal filtered at cuttoff freq

    USES : 
    same than lowpass, see :
    http://mpastell.com/2010/01/18/fir-with-scipy/
    """
    # N.B. in firwin, requencies are normalized to the Nyquist frequency, 
    # which is half the sampling rate !!
    freq_acq = 1.0/dt 

    freq_cutoff = cutoff_freq/(freq_acq/2) # cuttoff renormalized

    a=1 # filter denominator, in those cases simple convlution
    # -- On construit le filtre lineaire low pass !!
    b = signal.firwin(N_order,cutoff=freq_cutoff, #,width=freq_cutoff,\
                      window='hanning')
    # modification to have a high pass !!
    b = -b
    b[N_order/2]=b[N_order/2]+1
    return signal.lfilter(b,a,Signal0), b

def bandpass(Signal0,dt,min_freq,max_freq,N_order=int(1e3)):
    """
    argument : Signal0 (np.array()),
    dt : acquisistion time step (real) in SECOND
    cutoff_frequency (real) in HERZ
    N_order : order of the filter !! by default : 1000
    
    returns : np.array() signal filtered at between frequencies

    USES : 
    To get a bandpass FIR filter with SciPy we first need to 
    design appropriate lowpass and highpass filters and then combine them:
     see :
    http://mpastell.com/2010/01/18/fir-with-scipy/
    """
    freq_acq = 1.0/dt 
    min_freqR = min_freq/(freq_acq/2) # cuttoff renormalized
    max_freqR = max_freq/(freq_acq/2) # cuttoff renormalized

    a=1 # filter denominator, in those cases simple convlution
    # -- On construit le filtre lineaire low pass !!

    #Lowpass filter
    b1 = signal.firwin(N_order, cutoff = min_freqR, window = 'blackmanharris')
    #Highpass filter with spectral inversion
    b2 = - signal.firwin(N_order,cutoff=max_freqR, window = 'blackmanharris'); 
    b2[N_order/2] = b2[N_order/2] + 1
    #Combine into a bandpass filter 'd'
    d =-(b1+b2); 
    d[N_order/2] = d[N_order/2] + 1

    return signal.lfilter(d,a,Signal0), d


def smooth(x,window_len=11,window='hanning'):
    """
    smooth the data using a window with requested size.
    ============> source : scipy.org/Cookbook/SignalSmooth
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len/2-1):-(window_len/2)]


def poly_smooth(signal,x,degree=20,return_polynom=False):
    """
    smoothes a signal, using a polynomial approximation,
    the signal is integrated, then this integral is fitted by a polynom
    we derive the polynom analytically and the evaluation of this polynom
    should fit the intial signal,
    method taken from Claude Bedard, UNIC lab, Gif sur Yvette

    argument : signal=np.array(),x=np.array(),degree=int()

    returns : fitted_signal = np.array(), x_fitted = np.array(), poly=np.array()
    in case return_polynom=True, it returns the polynom
    N.B. the two arrays are smaller of one element
    """
    integral = integrate.cumtrapz(signal,x)
    poly_int = np.polyfit(x[:-1],integral,degree) # 1 point less after integration
    poly = np.polyder(poly_int)
    fitted_signal = np.polyval(poly,x[:-1])
    if return_polynom:
        return fitted_signal,x[:-1],poly
    else:
        return fitted_signal,x[:-1]


