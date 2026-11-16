"""Frozen common metrics; predictions in MW, capacity positive, all targets kept."""
import numpy as np
TAUS=np.array([.025,.05,.1,.25,.5,.75,.9,.95,.975])
LEVELS=[.5,.8,.9,.95]

def point(y,p,capacity):
 y=np.asarray(y).reshape(-1);p=np.asarray(p).reshape(-1)
 if y.shape!=p.shape:raise ValueError('Point prediction and target shapes differ')
 e=y-p;den=np.sum((y-np.mean(y))**2)
 return dict(n=len(y),rmse=float(np.sqrt(np.mean(e*e))),nrmse=float(np.sqrt(np.mean(e*e))/capacity),mae=float(np.mean(abs(e))),nmae=float(np.mean(abs(e))/capacity),r2=float(1-np.sum(e*e)/den) if den>0 else None)

def integrated_mean(q):
 q=np.sort(np.asarray(q),axis=1)
 if q.shape[1]!=9:raise ValueError('Fixed native nine-head grid required')
 # Piecewise linear quantile integral, constant at the two extreme heads.
 return np.trapz(np.c_[q[:,0],q,q[:,-1]],x=np.r_[0,TAUS,1],axis=1)

def interval(y,q,capacity,alpha=.05):
 q=np.sort(np.asarray(q),axis=1);lo=q[:,0];hi=q[:,-1];y=np.asarray(y)
 value=hi-lo+2/alpha*(np.maximum(lo-y,0)+np.maximum(y-hi,0));coverage=float(np.mean((lo<=y)&(y<=hi)))
 return dict(n=len(y),coverage=coverage,absolute_coverage_error=abs(coverage-(1-alpha)),width=float(np.mean(hi-lo)),nwidth=float(np.mean(hi-lo)/capacity),winkler=float(value.mean()),nwinkler=float(value.mean()/capacity))

def distribution(y,q,capacity):
 raw=np.asarray(q);q=np.sort(raw,axis=1)
 if q.shape[1]!=9:raise ValueError('Nine heads required')
 parts={};weighted=.5*abs(np.asarray(y)-q[:,4])
 for coverage in LEVELS:
  alpha=1-coverage;i=int(np.argmin(abs(TAUS-alpha/2)));j=int(np.argmin(abs(TAUS-(1-alpha/2))))
  z=interval(y,q[:,[i,j]],capacity,alpha);parts[str(coverage)]=z
  lo=q[:,i];hi=q[:,j];weighted+=alpha/2*(hi-lo+2/alpha*(np.maximum(lo-y,0)+np.maximum(y-hi,0)))
 pin=np.maximum(TAUS[None,:]*(np.asarray(y)[:,None]-raw),(TAUS[None,:]-1)*(np.asarray(y)[:,None]-raw))
 return dict(n=len(y),wis=float(np.mean(weighted)/4.5),nwis=float(np.mean(weighted)/4.5/capacity),mean_pinball=float(pin.mean()),raw_crossing_fraction=float(np.mean(np.any(np.diff(raw,axis=1)<0,axis=1))),levels=parts,median=point(y,q[:,4],capacity),integrated_mean=point(y,integrated_mean(q),capacity))

def expand_calibration(ycal,qcal,qeval,coverage=.95):
 alpha=1-coverage;qc=np.sort(qcal,axis=1);qe=np.sort(qeval,axis=1);scores=np.maximum.reduce([qc[:,0]-ycal,ycal-qc[:,-1],np.zeros(len(ycal))]);rank=int(np.ceil((len(scores)+1)*coverage))
 if rank>len(scores):raise ValueError('Finite calibration rank unavailable')
 delta=float(np.partition(scores,rank-1)[rank-1]);return np.c_[qe[:,0]-delta,qe[:,-1]+delta],dict(delta=delta,rank=rank,n=len(scores),coverage=coverage)
