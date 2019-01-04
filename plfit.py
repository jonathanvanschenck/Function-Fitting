import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def vecAnd(listArrays):
    return np.all(np.array(listArrays).T,axis=1)
def conv(a,b,dx=1):
    return np.fft.irfft(np.fft.rfft(a)*np.fft.rfft(b))*dx
def mexp(x,p):
    res = np.zeros(len(x))
    for i in range(len(p)//2):
        res=res+p[2*i]*np.exp(-x/p[2*i+1])
    return res
def mexp2(x,p):
    res = np.zeros(len(x))
    for i in range(len(p)//2):
        res=res+np.exp(p[2*i]-x/p[2*i+1])
    return res
def unitStep(t):
    if t>=0:
        return 1
    else:
        return 0
def unitStepVec(t):
    return np.vectorize(unitStep)(t)
def rhoT(x,p,tau):
    capR = np.zeros(len(x))
    rho = np.zeros(len(x))
    nrho = np.zeros(len(x))
    for i in range(len(p)//2):
        capR += p[2*i]*np.exp(-x/p[2*i+1])/(np.exp(tau/p[2*i+1])-1)
        rho += unitStepVec(x)*p[2*i]*np.exp(-(x)/p[2*i+1])
        nrho += unitStepVec(x-tau)*p[2*i]*np.exp(-(x-tau)/p[2*i+1])
    return capR+rho+nrho
def rhoT2(x,p,tau):
    capR = np.zeros(len(x))
    rho = np.zeros(len(x))
    nrho = np.zeros(len(x))
    for i in range(len(p)//2):
        capR += np.exp(p[2*i]-x/p[2*i+1])/(np.exp(tau/p[2*i+1])-1)
        rho += unitStepVec(x)*np.exp(p[2*i]-(x)/p[2*i+1])
        nrho += unitStepVec(x-tau)*np.exp(p[2*i]-(x-tau)/p[2*i+1])
    return capR+rho+nrho
def rot(x,s):
    l = len(x)
    step = s%l
    if step!=0:
        return x[np.r_[step:l,0:step]]
    return x

class FIT:
    """    
    Inputs Required
    datat:      Some generic time data to be fit (t values in ns)
    datay:      Some generic data to be fit (y values)
    expCoef:    Boonlean to specify to use exponential coefficients
    
    Attributes: 
    paramNames: 1d Numpy array holding the names of each parameter
    iparam:     1d numpy array holding the initial guess for paramter values
                 (must be specified using .initializeFitParams BEFORE the fit
                  can be performed). Structure: [...]
    param:      1d numpy array holding the resulting parameter values after 
                 fitting. Structure: [...]
    which:      1d bool array holding specifying which of the parameters will
                 be allowed to varrying during fitting. Default is to allow all
                 parameters to varry. Can be modified using .freezeFitParams().
                 Structure: [...?]
    bound:      2d numpy array holding the paramater bounds to be used during 
                 fitting. If parameters have been frozen by using .freezeFitParam
                 method, then the bounds provided here will be ignored during 
                 fitting. So, bound.shape[0]=param.shape[0].
                  [[..._],
                   [...^]]
    nf:         Value holding the number of parameters allowed to varry
    fitMask:    1d bool array specifying which data points to be 
                 used during fitting. Default is to use all data.
                 Can be modified using .createFitRegion()
                 
    
    Best Practice for Use:
         1a)  Call fit class and provide data
    opt  1b)  Provide IRF to fit with (.loadIRF)
    opt   2)  Specify fit region (.createFitRegion)
          3)  Provide inital guess for parameter values (.initializeFitParams)
    opt   4)  Freeze parameters NOT used during fitting (.freezeFitParams)
          5)  Set bounds on free fit parameters (.createFitParamBounds)
          6)  Perform Fit (.performFit)
     opt/ 7)  Plot resuts (.plot)
     opt\ 8)  Print fit results (.printParam)
    """
    def __init__(self,datat,datay,expCoef=False):
        """
        datat:       Some generic numpy array to be fit (x values). Must match length of datay
        datay:       Some generic numpy array to be fit (y values). Must match length of datat
        fun:         Function to be fit to. It must only take a single argument (param),
                      and then return a numpy array of y values of the fit. Must match length
                      of datay.
        paramNames:  Numpy array holding the names of each parameter used by fun to fit datay
        """
        self.dt = np.mean(np.diff(datat))
        self.datat = datat-np.min(datat)
        self.tau = np.max(self.datat)
        self.datay = datay
        self._expCoef = expCoef
        if self._expCoef:
            self.fun = lambda p: rhoT2(self.datat-p[-1],p[:-1],self.tau)
            self.paramNames = np.array(["ea1","t1","s"])#np.vectorize(str)(np.array(paramNames))
        else:
            self.fun = self.fun = lambda p: rhoT(self.datat-p[-1],p[:-1],self.tau)
            self.paramNames = np.array(["a1","t1","s"])#np.vectorize(str)(np.array(paramNames))
        self.iparam = np.zeros(len(self.paramNames))
        self.param = np.copy(self.iparam)
        self.which = np.full(len(self.paramNames),True,dtype='bool')
        self.nf = np.sum(np.ones(len(self.paramNames))[self.which])
        self.fitMask = np.full(len(self.datat),True,'bool')
        self.bound = np.array((2,len(self.paramNames)))
        self.irf = np.array([1/self.dt]+(len(datat)-1)*[0])
        self.irffit = np.array([1/self.dt]+(len(datat)-1)*[0])
        
    def loadIRF(self,irf):
        self.irf = irf/np.sum(irf*self.dt)
        imax = np.arange(len(irf))[irf==np.max(irf)][0]
        self.irffit = rot(self.irf,imax)
    
    def createFitRegion(self,mini,maxi):
        """
        Function creates a boolean mask for data to be use during fitting to
        select which points to fit by.
        
        Input
        mini:       Float: left bound on datat range
        maxi:       Float: right bound on datat range
        """
        self.fitMask = vecAnd([self.datat<maxi,self.datat>mini])
                    
    def freezeFitParams(self,which):
        """
        Function allows user to freeze particular parameters during fitting. 
        By specifying boolean "False" for a parameter, it is not allowed to vary
        during fitting. Structure: [...?]
        
        Input
        which:      1d Boolean list/array. Array MUST be the same length as 
                     iparam/param/paramNames. If which[i]==True, than param[i]
                     is allowed to vary during fitting. If which[i]==False, 
                     than param[i] is frozen during fitting.
        """
        self.which = np.array(which,dtype='bool')
        self.nf = np.sum(np.ones(len(self.paramNames))[self.which])
    
    def initializeFitParams(self,iparam):
        """
        Function sets the initial guess for parameter values
        
        Input
        iparam:     1d array/list which holds the initial guess for each
                     parameter value. The length MUST match .paramNames. Structure:
                     [...]
        """
        self.iparam = np.array(iparam)
        self.param = np.copy(self.iparam)
        if self._expCoef:
            self.paramNames = np.array([i+str(j+1) for j in range((len(iparam)-1)//2) for i in ["ea","t"]]+["s"])
        else:
            self.paramNames = np.array([i+str(j+1) for j in range((len(iparam)-1)//2) for i in ["a","t"]]+["s"])
        self.which = np.full(len(self.paramNames),True,dtype='bool')
        
    def createFitParamBounds(self,bound):
        """
        Function sets the bounds for parameter values during fitting
        
        Input
        bound:      2d array/tuple holding the paramater bounds to be used during 
                     fitting. If parameters have been frozen by using .freezeFitParam
                     method, then bound values will be ignored by fitting proceedure.
                     The length of each piece MUST match .iparam/.param/.paramNames,
                     i.e. bound.shape[0]=bound.shape[1]=.iparam.shape[0]=...
                         [[..._],
                          [...^]]
        """
        self.bound = np.array(bound)
    
    def fitFun(self,par):
        """
        Function to fit to data. Takes in the free parameters and uses
        .fun to generate the y-points to fit.
        
        Input
        par:        1d numpy array holding FREE parameters to be modified
                     during fitting
        
        Output
        res:        1d numpy array holding generated y-points
        
        """
        p = np.copy(self.param)
        p[self.which] = np.array(par)
        res = conv(self.fun(p),self.irffit,self.dt)
        while res.shape[0]<self.datat.shape[0]:
            res = np.hstack([res,[0]])
        return res
        
    def fitFunDifference(self,par):
        """
        Function gives the error between fit and data. Used by 
        scipy.optimize.least_squares to minimize the SSE.
        
        Input:
        par:        1d numpy array holding FREE parameters to be modified
                     during fitting
                     
        Output
        res:        1d numpy array holding the error between fit and datay.
        """
        d = self.fitFun(par)[self.fitMask]
        y = self.datay[self.fitMask]
        mask = vecAnd([d>0.0,y>0.0])
        res = np.zeros(len(y))
        res[mask] = np.log10(d[mask])-np.log10(y[mask])
        return res
    
    def plot(self,plotName='',xlab='',ylab='',log=True):
        """
        Function gives a plot of the data and fit. Must be called AFTER 
        .initializeFitParams, but can be called before .performFit.
        
        Input
        plotName:   String which titles the plot.
        """
        plt.figure()
        if log:
            plt.yscale('log')
        else:
            plt.yscale('linear')
        data, = plt.plot(self.datat,self.datay,'o',label='Data')
        fit, = plt.plot(self.datat,self.fitFun(self.param[self.which]),label='fit ex',color='cyan')
        fit2, = plt.plot(self.datat[self.fitMask],self.fitFun(self.param[self.which])[self.fitMask],label='fit2',color='red')
        plt.title(plotName)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.ylim(ymin=np.min(self.datay[self.datay>0]))
        plt.legend()#(loc='lower right')
        #plt.show()
    
    def _normalizeP(self,p):
        """
        Function normalizes provided parameter values on 0..1 for the range 
        provided by .bound
        
        Input:
        p:        1d Numpy array holding parameter values. MUST match .param
                   in length
        
        Output:
        res:    ` 1d numpy array holding values between 0 to 1 representing
                   where in the range .bound[0][i] to .bound[1][i] the value
                   p[i] resides.
        """
        return (p-self.bound[0])/(self.bound[1]-self.bound[0])
        
    def performFit(self,xtol=3e-16,ftol=1e-10,num=6):
        """
        Function modifies param[which] so as to minimize the SSE using
        scipy.optimize.least_squares.
        
        Input
        xtol:       See least_squares documentation
        ftol:       See least_squares documentation
        num:        Integer holding the number of parameters to be printed on
                     each line
                     
        Output
        res:        Prints out "Start" iparam[which], "End" param[which] and 
                     "Shift" (param-iparam)[which] as a percentage of upper and
                     lower bounds. This is used to see if any parameters have 
                     "hit" the edges of their range during fitting. This can be
                     seen by as "End" being either 0.0 or 1.0. "Start" can be 
                     used to see if the bounds are too loose, or too strict.
                     And "Shift" gives a sense for how good the initial guess
                     was.
        """
        self.fit = least_squares(self.fitFunDifference,self.iparam[self.which],
                                 verbose=1,bounds=self.bound[...,self.which],xtol=xtol,ftol=ftol)
        if self.fit.success:
            self.param = np.copy(self.iparam)
            self.param[self.which] = np.copy(self.fit.x)
        else:
            print('Fit Falue, see: self.fit.message')
            self.param = np.copy(self.iparam)
        start = self._normalizeP(self.iparam)[self.which]#(self.iparam[self.which]-np.array(self.bound[0]))/(np.array(self.bound[1])-np.array(self.bound[0]))
        end = self._normalizeP(self.param)[self.which]#(self.param[self.which]-np.array(self.bound[0]))/(np.array(self.bound[1])-np.array(self.bound[0]))
        difference = end-start#(self.param[self.which]-self.iparam[self.which])/(np.array(self.bound[1])-np.array(self.bound[0]))
        st = lambda x: '{0:6.3f}'.format(x)
        st2 = lambda x: "{0:>6}".format(x)
        if len(self.paramNames[self.which])%num==0:
            setp = np.arange(len(self.paramNames[self.which])//num)
        else:
            setp = np.arange((len(self.paramNames[self.which])//num)+1)
        for i in setp:
            print(np.hstack([np.array([[' Name'],['Start'],['  End'],['Shift']]),
                             np.vstack([np.vectorize(st2)(self.paramNames[self.which][(num*i):(num*(i+1))]),
                                       np.vectorize(st)(start[(num*i):(num*(i+1))]),
                                       np.vectorize(st)(end[(num*i):(num*(i+1))]),
                                       np.vectorize(st)(difference[(num*i):(num*(i+1))])])
                            ]))     
                                
    def printParam(self,num=6):
        """
        Function prints out the parameter values and names.
        
        Input
        num:        Integer specifying the number of parameters to print onto
                     each line
        """
        st = lambda x: "{0:6.3f}".format(x)
        st2 = lambda x: "{0:>6}".format(x)
        if len(self.paramNames)%num==0:
            setp = np.arange(len(self.paramNames)//num)
        else:
            setp = np.arange((len(self.paramNames)//num)+1)
        for i in setp:
            print(np.hstack([[[' Name'],['Value']],
                             np.vstack([
                                        np.vectorize(st2)(self.paramNames[(num*i):(num*(i+1))]),
                                        np.vectorize(st)(self.param[(num*i):(num*(i+1))])
                                        ])
]))
