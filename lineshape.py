import numpy as np

from numpy import pi

import matplotlib.pyplot as plt

from scipy.optimize import least_squares

def Glorz(x,mu,sig):
    return sig/((2*np.pi)*((x-mu)**2+(sig/2)**2))

def Gauss(x,mu,doubsig):
    sig = doubsig/2
    return (1/(sig*np.sqrt(2*np.pi)))*np.exp(-(x-mu)**2/(2*sig**2))

def vec(list1d):
    "Takes in list [i1,i2,...] and returns a numpy.array column"
    return np.array(list1d,ndmin=2).T

def cbind(listVectors):
    "Takes in a list of numpy.array columns and/or 2d numpy.arrays, and\
     combines columnwise"
    return np.hstack(listVectors)
    
def colorMapR(i):
    "Returns the name of an R-like indexed color map, has 10 colors"    
    cMapping = ['black','red','blue','green','purple','yellow',
                'magenta','orange','brown','cyan']
    return cMapping[i%10]
    
def vecAnd(listArrays):
    return np.all(np.array(listArrays).T,axis=1)

def unitStep(t):
    if t>=0:
        return 1
    else:
        return 0

def unitStepVec(t):
    return np.vectorize(unitStep)(t)

def conv(a,b,dx=1):
    return np.fft.irfft(np.fft.rfft(a)*np.fft.rfft(b)*dx)
    
def rjust(string,wid,fill=' '):
    return string.rjust(wid,fill)

def vecNot(vect):
    return np.array([not i for i in vect],dtype='bool')

from math import factorial

def overlap(m,s):
    return np.exp(-s)*(s**m)/factorial(m)

vecOverlap = np.vectorize(overlap)

def rangeNot2(low,high,n):
    res = np.arange(0,high-low+1)+low
    return res[res != n]

def bandshift(m,s,Nmax=10):
    return np.sum(vecOverlap(rangeNot2(0,Nmax,m),s)/(rangeNot2(0,Nmax,m)-m))

def Wcontrib(m,s,W,Nmax=10):
    return (1-0.5*W*bandshift(m,s,Nmax))**2

def intDiscrete(x,y):
    dx=x[1:]-x[:-1]
    dx = np.append(dx,dx[-1])
    return np.sum(dx*y)

def absLineMu2(x,p,n,fun=Glorz):
    #p=area,deltaSig/sig,HR fact,Intermol. Coupling,E00,Evib,sig
    #   0         1         2       3                4   5    6  
    dx = np.absolute(np.mean(x[1:]-x[:-1]))
    x2 = np.arange(p[4]-5*p[5],p[4]+n*p[5]+5*p[5],dx)
    res = overlap(0,p[2])*Wcontrib(0,p[2],4*p[3])*fun(x,p[4]+(2*p[3]*p[6]*overlap(0,p[1])),p[6])
    res2 = overlap(0,p[2])*Wcontrib(0,p[2],4*p[3])*fun(x2,p[4]+(2*p[3]*p[6]*overlap(0,p[1])),p[6])
    for m in np.arange(1,n):
        res = res+overlap(m,p[2])*Wcontrib(m,p[2],4*p[3])*fun(x,p[4]+(2*p[3]*p[6]*overlap(m,p[2]))+m*p[5],(1+m*p[1])*p[6])
        res2 = res2+overlap(m,p[2])*Wcontrib(m,p[2],4*p[3])*fun(x2,p[4]+(2*p[3]*p[6]*overlap(m,p[2]))+m*p[5],(1+m*p[1])*p[6])
    normConst = intDiscrete(x2,res2)
    return p[0]*res/normConst
