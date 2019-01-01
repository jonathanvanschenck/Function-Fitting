import Fit

def unitStep(t):
    if t>=0:
        return 1
    else:
        return 0

def unitStepVec(t):
    return np.vectorize(unitStep)(t)

def conv(a,b):
    return np.fft.irfft(np.fft.rfft(a)*np.fft.rfft(b))

class plLifeFit:
    "Creates a PL Lifetimes Fitting object"
    def __init__(self,photonCounts,dt):
        self.dt = dt
        self.n = photonCounts.shape[0]
        self.t = dt*np.arange(0,photonCounts.shape[0])
        self.countsMax = max(photonCounts)
        self.countsMaxTime = (self.t[photonCounts == self.countsMax])[0]
        self.counts = photonCounts/self.countsMax
        self.countsMask = np.full(self.counts.shape[0],True,dtype=bool)
        #self.nonZeroCounts = self.counts>1e-6 #TimeHarp only goes upto 1e6
        self.nonZeroCounts = photonCounts>0
        self.logCounts = np.log10(np.full(self.counts.shape[0],1e-6))
        self.logCounts[self.nonZeroCounts] = np.log10(self.counts[self.nonZeroCounts])
        self.IRF = np.hstack([np.array([1]),np.zeros(self.counts.shape[0]-1)])
        self.IRFMax = 1
        self.IRFNorm = self.IRF.copy()
        self.numTimeScales = 1
        #self.rescaling = np.array([1])
        #self.paramInitial = np.array([3.,0.,1.])
        self.initalizeParameters(3)
        self.param = self.paramInitial
        self.convertParameters()
        self.shifty = 100
        self.createFitRegion(self.countsMaxTime-.1,self.countsMaxTime+5)

    def attachSpectrum(self,wavelengths,spectrum,master=False):
        self.nm = wavelengths
        self.specMask = np.full(wavelengths.shape[0],True,dtype=bool)
        if master:
            self.specMaster = spectrum
        else:
            self.spec = spectrum

    def attachIRF(self,IRF):
        self.IRFMax = max(IRF)
        self.IRF = IRF/self.IRFMax
        self.IRFNorm = self.IRF/(np.sum(self.IRF[0:-1]*np.diff(self.t)))
    
    def createSpectrumMask(self, minimum,maximum):
        self.specMask = vecAnd([self.nm > minimum, self.nm < maximum])
        
    def createCountsMask(self, minimum,maximum):
        self.countsMask = vecAnd([self.t > minimum, self.t < maximum])
    
    def plotSpec(self,master=False,unmask=False,legend=True,add=False):
        if unmask:
            mask = np.full(self.nm.shape[0],True,dtype=bool)
        else:
            mask = self.specMask
        if not add:
            plt.figure()
        if master:
            plt.plot(self.nm[mask],self.specMaster[mask],label='FullSpec')
        plt.plot(self.nm[mask],self.spec[mask],label='Spectrum')
        if legend and not add:
            plt.legend()
        if not add:
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('PL (counts)')
            #plt.show()
        
    def plot(self,lab='',log=True,noFit=False,unmask=False,withSpec=False):
        if unmask:
            mask = np.full(self.counts.shape[0],True,dtype=bool)
        else:
            mask = self.countsMask
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        countsLine, = plt.plot(self.t[mask],self.counts[mask],'o',
                               color='black',label='Data')
        IRFLine, = plt.plot(self.t[mask],self.IRF[mask],#'o',
                            color='blue',label='IRF')
        plt.ylim(ymin=min(self.counts[vecAnd([mask,self.counts>1e-5])]))
        plt.ylim(ymax=1.1)
        if not noFit:
            fitLine, = plt.plot(self.t[mask],
                                   (self.fitFun2(self.param))[mask],
                                   color='red',label='Fit')
        if log:
            ax.set_yscale('log')
            plt.ylabel('Log Counts (Normalized)')
        else:
            plt.ylabel('Counts (Normalized)')
        plt.xlabel('Time (ns)')
        plt.title(lab)
        plt.legend()
        plt.show()
            #if withSpec:
            #    plt.axes([.8,.8,.9,.9])
            #    self.plotSpec(master=True,legend=False)
                
    def createFitRegion(self, minimum,maximum):
        self.fitMask = vecAnd([self.t > minimum, self.t < maximum])
        self.fitMaskMin = minimum
        self.fitMaskMax = maximum
    
    def createNumTimeScales(self,n,rescal=1):
        self.numTimeScales = n
        if rescal==1:
            self.rescaling = np.ones(n)
        else:
            self.rescaling = np.array(rescal)
        
    def fitFun(self,a,tau,shift,totalScale=1):
        exponential = np.zeros(self.n)
        for i in np.arange(0,self.numTimeScales):
            exponential += unitStepVec(self.t-shift-self.shifty*self.dt)*self.rescaling[i]*a[i]*np.exp(-(self.t-shift-self.shifty*self.dt)/tau[i])
        result = conv(self.IRFNorm[(np.arange(0,self.n)+self.shifty)%self.n],exponential)
        #result = exponential
        while (result.shape[0] < self.n):
            result = np.append(result,[0])
        return totalScale*result/np.max(result)
    
    def fitFun2(self,param):
        if   self.numTimeScales == 1:
            return self.fitFun([1],[param[0]],param[1],totalScale=param[2])
        elif self.numTimeScales == 2:
            return self.fitFun(np.hstack([[1],[param[0]]]),
                                param[(self.numTimeScales-1):(2*self.numTimeScales-2+1)],
                                param[2*self.numTimeScales-1],
                                param[2*self.numTimeScales])
        else:
            return self.fitFun(np.hstack([[1],param[0:(self.numTimeScales-2+1)]]),
                                param[(self.numTimeScales-1):(2*self.numTimeScales-2+1)],
                                param[2*self.numTimeScales-1],
                                param[2*self.numTimeScales])
                                
    def fitFunDifference(self,param):
        d = self.fitFun2(param)
        mask = vecAnd([self.fitMask,d>0,self.nonZeroCounts])
        return (np.log10(d[mask])-self.logCounts[mask])
        #return np.dot(d[mask]-self.logCounts[mask],d[mask]-self.logCounts[mask])
     
    def initalizeParameters(self,tau,a=1,shift=0,totScale=1,numTimeScales=1,rescaling=1):
        self.createNumTimeScales(n=numTimeScales,rescal = rescaling)
        self.paramInitial = np.zeros(2*self.numTimeScales+1)
        if   numTimeScales==1:
               self.paramInitial[0] = tau
               self.paramInitial[1] = shift
               self.paramInitial[2] = totScale
        elif numTimeScales==2:
               self.paramInitial[0] = a
               self.paramInitial[1:3] = np.array(tau)
               self.paramInitial[3] = shift
               self.paramInitial[4] = totScale
        else:
               self.paramInitial[0:(self.numTimeScales-2+1)] = np.array(a)
               self.paramInitial[(self.numTimeScales-1):(2*self.numTimeScales-2+1)] = np.array(tau)
               self.paramInitial[2*self.numTimeScales-1] = shift
               self.paramInitial[2*self.numTimeScales] = totScale

    def convertParameters(self):
        if   self.numTimeScales==1:
               self.a = self.rescaling[0]
               self.tau = self.param[0]
               self.shift = self.param[1]
               self.totScale = self.param[2]
        elif self.numTimeScales==2:
               self.a = np.array([1]+[self.param[0]])*self.rescaling
               self.tau = self.param[1:3]
               self.shift = self.param[3]
               self.totScale = self.param[4]
        else:
               self.a = self.param[0:(self.numTimeScales-2+1)]*self.rescaling
               self.tau = self.param[(self.numTimeScales-1):(2*self.numTimeScales-2+1)]
               self.shift = self.param[2*self.numTimeScales-1]
               self.totScale = self.param[2*self.numTimeScales]   

    def performFit(self,bound,xtol=3e-16,ftol=1e-10):
        self.fit = least_squares(self.fitFunDifference,self.paramInitial,
                                 verbose=1,bounds=bound,xtol=xtol,ftol=ftol)
        if self.fit.success:
            self.param = self.fit.x
        else:
            print('Fit Falue, see: self.fit.message')
            self.param = self.paramInitial
        self.convertParameters()
        print('Initial Parameter Location in Bound')
        print((self.paramInitial-np.array(bound[0]))/(np.array(bound[1])-np.array(bound[0])))
        print('Final Parameter Location in Bound')        
        print((self.param-np.array(bound[0]))/(np.array(bound[1])-np.array(bound[0])))
        print('Shift')
        print((self.param-self.paramInitial)/(np.array(bound[1])-np.array(bound[0])))

    def printParams(self):
#        if   self.numTimeScales==1:
#               a = self.rescaling[0]
#               tau = self.param[0]
#               shift = self.param[1]
#               totScale = self.param[2]
#        elif self.numTimeScales==2:
#               a = np.array([1]+[self.param[0]])*self.rescaling
#               tau = self.param[1:3]
#               shift = self.param[3]
#               totScale = self.param[4]
#        else:
#               a = self.param[0:(self.numTimeScales-2+1)]*self.rescaling
#               tau = self.param[(self.numTimeScales-1):(2*self.numTimeScales-2+1)]
#               shift = self.param[2*self.numTimeScales-1]
#               totScale = self.param[2*self.numTimeScales]
        print('Characteristic Timescales (ns)')
        print(self.tau)#print(tau)
        print('Proportional Weights')
        print(self.a/np.sum(self.a))#print(a/np.sum(a))
        print('Total Scale Factor: '+str(self.totScale))#print('Total Scale Factor: '+str(totScale))
        print('Time Shift: '+str(self.shift)+' ns')#print('Time Shift: '+str(shift)+' ns')
        
