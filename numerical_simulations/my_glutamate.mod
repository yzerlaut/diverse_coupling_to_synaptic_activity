COMMENT
//****************************//
// Created by Alon Polsky 	//
//    apmega@yahoo.com		//
//		2007			//
//****************************//

slightly modified by Y. Zerlaut (yann.zerlaut@gmail.com)
to include a net_receive and be stimulated by a shotnoise, 2016
ENDCOMMENT

TITLE NMDA synapse with depression

NEURON {
	POINT_PROCESS my_glutamate
	USEION ca READ cai WRITE ica VALENCE 2
	NONSPECIFIC_CURRENT inmda,iampa
	RANGE e ,gmax,ntar,local_v,inmda,iampa,gh
	RANGE del,Tspike, Nspike, nmda_on
	RANGE gnmda, gampa,tau_ampa
	GLOBAL n, gama,tau1,tau2,tauh,cah
}

UNITS {
	(nA) 	= (nanoamp)
	(mV)	= (millivolt)
	(nS) 	= (nanomho)
	(mM)    = (milli/liter)
        F	= 96480 (coul)
        R       = 8.314 (volt-coul/degC)

}

PARAMETER {
	gmax=1	(nS)
	e= 0.0	(mV)
	tau1=90	(ms)	
	tau2=5	(ms)	
	tau_ampa=2	(ms)	
	n=0.25 	(/mM)	
	gama=0.08 	(/mV) 
	dt (ms)
	ntar=.3	:NMDA to AMPA ratio
	v		(mV)
	del=30	(ms)
	Tspike=10	(ms)
	nmda_on=1
	cah   = 10	(/ms)		: max act rate  
	tauh   = 1000	(/ms)		: max deact rate 
}

ASSIGNED { 
	inmda		(nA)  
	iampa		(nA)  
	gnmda		(nS)
	gh		(nS)
	ica 		(mA/cm2)
	local_v	(mV):local voltage
	cai		(mM)	
}
STATE {
	A (nS)
	B (nS)
	gampa
	h		(nS)

}

INITIAL {
      gnmda=0 
      gampa=0 
	h=0
	A=0
	B=0

}    

BREAKPOINT {  
    
	SOLVE state METHOD cnexp

	gnmda=(A-B)/(1+n*exp(-gama*v) )
	gh=(exp(-h))
	inmda =(1e-3)* gnmda * gh * (v-e) * nmda_on
	ica=inmda/10
	iampa= (1e-3)*gampa* (v- e)
	local_v=v
}

DERIVATIVE state {
	A'=-A/tau1
	B'=-B/tau2
	gampa'=-gampa/tau_ampa
	h'=(cah*cai-h)/tauh
}

NET_RECEIVE(weight (uS)) {
	gampa = gampa + gmax
	A = A + gmax*ntar
	B = B + gmax*ntar
}


