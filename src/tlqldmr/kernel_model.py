"""Matched fixed-feature TL-QLDMR/SVQR for the full formal experiment runner.

All y and outputs retain the supplied label units.  The owning runner is
responsible for fit-only preprocessing and returning predictions to MW.
Features/landmarks use fit covariates only; no selection or scoring happens here.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import rbf_kernel
from .structured_ipm import solve
TAUS=np.array([.025,.05,.1,.25,.5,.75,.9,.95,.975])

def digest(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()

def fit_distance_scale(X,seed=20261113,pairs=4096):
    rng=np.random.default_rng(seed);ix=rng.integers(0,len(X),size=(pairs,2))
    d=np.sum((X[ix[:,0]]-X[ix[:,1]])**2,axis=1);positive=d[d>0]
    if not len(positive):raise ValueError('All sampled fit distances are zero')
    return {'median_squared_distance':float(np.median(positive)),
            'pair_seed':seed,'pairs':pairs,'nonzero_pairs':len(positive)}

def candidate_schedule(X,ns,nt,*,count=24,seed=20261113,ranks=(128,256)):
    """24 distinct base configurations; p/eta1/eta2 use independent LHS axes.

    The same C/gamma/p/rank/landmark rows must be offered to matched SVQR.
    Component interventions belong to the later paired ablation, not duplicate
    base configurations that would spend SVQR's search budget twice.
    """
    if count<24:raise ValueError('The full schedule requires at least 24 queries')
    u=qmc.LatinHypercube(d=8,seed=seed).random(count)
    scale=fit_distance_scale(X,seed=seed);dm=scale['median_squared_distance'];n=ns+nt
    ps=np.array([nt/n,.25,.5,.75,.9,.95,.99]);rows=[]
    for i,v in enumerate(u):
        C=float(10**(-2+4*v[0]));e1=float(10**(-7+6*v[3]));e2=float(10**(-7+6*v[4]))
        p=float(ps[min(int(v[2]*len(ps)),len(ps)-1)])
        row=dict(candidate=i,C=C,gamma=float(10**(-1+2*v[1])/dm),target_fraction=p,
            eta1=e1,eta2=e2,lambda1=e1*C*n,lambda2=e2*C*n,
            rank=int(ranks[min(int(v[5]*len(ranks)),len(ranks)-1)]),
            landmark_strategy=['uniform','balanced_domains'][min(int(v[6]*2),1)],
            regularization_parameterization='effective',label='independent_LHS')
        # Prespecified zero/single-term controls count inside the search budget.
        if i in [4,8,12,16,20]:row.update(eta1=0.,lambda1=0.,label='zero_variance_LHS_base')
        if i in [6,12,18]:row.update(eta2=0.,lambda2=0.,label='zero_alignment_LHS_base')
        rows.append(row)
    rows[0].update(C=50.,gamma=.002,target_fraction=.99,eta1=.0005/(50*n),eta2=.0001/(50*n),
        lambda1=.0005,lambda2=.0001,rank=256,landmark_strategy='uniform',label='original_reference')
    rows[1].update(C=1.,gamma=1/dm,target_fraction=nt/n,eta1=0.,eta2=0.,lambda1=0.,lambda2=0.,
        rank=256,landmark_strategy='balanced_domains',label='empirical_mass_zero_terms')
    base_keys=[(r['C'],r['gamma'],r['target_fraction'],r['rank'],r['landmark_strategy']) for r in rows]
    assert len(set(base_keys))==count,'SVQR must receive distinct base configurations'
    return rows,scale

@dataclass
class FeatureMap:
    components:np.ndarray
    normalization:np.ndarray
    gamma:float
    original_indices:np.ndarray
    info:dict
    def transform(self,V):
        V=np.asarray(V,dtype=float)
        if not len(V):return np.empty((0,len(self.components)))
        return rbf_kernel(V,self.components,gamma=self.gamma)@self.normalization.T


def prepare_features(X,ns,params,seed):
    X=np.asarray(X,dtype=float);n=len(X);rank=min(int(params['rank']),n)
    rng=np.random.RandomState(seed);strategy=params.get('landmark_strategy','uniform')
    if strategy=='uniform':indices=rng.permutation(n)[:rank]
    elif strategy=='balanced_domains':
        nt=n-ns
        if ns<=0 or nt<=0:indices=rng.permutation(n)[:rank]
        else:
            takeS=min(ns,rank//2);takeT=min(nt,rank-takeS)
            takeS=min(ns,rank-takeT)
            indices=np.r_[rng.permutation(ns)[:takeS],ns+rng.permutation(nt)[:takeT]]
            if len(indices)<rank:
                remaining=np.setdiff1d(np.arange(n),indices,assume_unique=False)
                indices=np.r_[indices,rng.permutation(remaining)[:rank-len(indices)]]
    else:raise ValueError('Unknown landmark strategy: '+strategy)
    start=time.perf_counter()
    # Fitting on the preselected landmark set uses the same sklearn numerical
    # normalization rule; all selected rows are training inputs.
    mapper=Nystroem(kernel='rbf',gamma=params['gamma'],n_components=rank,random_state=seed)
    mapper.fit(X[indices]);actual=indices[mapper.component_indices_]
    fm=FeatureMap(mapper.components_.copy(),mapper.normalization_.copy(),float(params['gamma']),actual.copy(),{})
    Z=fm.transform(X)
    info=dict(rank=rank,strategy=strategy,seed=int(seed),source_landmarks=int(np.sum(actual<ns)),
              target_landmarks=int(np.sum(actual>=ns)),input_sha256=digest(X),feature_sha256=digest(Z),
              landmark_indices_sha256=digest(actual),normalization_sha256=digest(fm.normalization),
              construction_seconds=time.perf_counter()-start)
    fm.info=info
    return fm,Z,info


def objective(method,Z,y,ns,nt,params):
    n=len(y);assert ns+nt==n and ns>=0 and nt>=0
    C=float(params['C']);p=float(params['target_fraction']);m=Z.shape[1]
    if not 0<=p<=1 or C<=0:raise ValueError('Invalid C or domain mass')
    if ns==0 or nt==0:weights=np.ones(n)
    else:weights=np.r_[np.full(ns,n*(1-p)/ns),np.full(nt,n*p/nt)]
    mode=params.get('regularization_parameterization','nominal')
    if mode=='effective':l1=float(params['eta1'])*C*n;l2=float(params['eta2'])*C*n
    elif mode=='nominal':l1=float(params.get('lambda1',0.));l2=float(params.get('lambda2',0.))
    else:raise ValueError('Unknown regularization parameterization')
    if method in ['Standard-SVQR','Weighted-SVQR']:l1=l2=0.
    elif method=='NoVariance':l1=0.
    elif method=='NoAlignment':l2=0.
    elif method!='TL-QLDMR':raise ValueError('Unknown kernel method: '+method)
    if min(l1,l2)<0:raise ValueError('Penalty coefficients must be nonnegative')
    R=np.eye(m);c=np.zeros(m)
    if ns>0 and l1>0:
        centered=Z[:ns]-Z[:ns].mean(0);yc=y[:ns]-y[:ns].mean()
        R+=l1*(centered.T@centered)/ns;c+=l1*(centered.T@yc)/ns
    if ns>0 and nt>0 and l2>0:
        delta=Z[:ns].mean(0)-Z[ns:].mean(0);R+=l2*np.outer(delta,delta)
    return R,c,C*weights,dict(lambda1=l1,lambda2=l2,eta1=l1/(C*n),eta2=l2/(C*n),
        source_loss_mass=float(weights[:ns].sum()),target_loss_mass=float(weights[ns:].sum()),
        parameterization=mode)

@dataclass
class FittedKernel:
    feature_map:FeatureMap
    W:np.ndarray
    b:np.ndarray
    duals:np.ndarray
    taus:np.ndarray
    R:np.ndarray
    c:np.ndarray
    loss_coefficients:np.ndarray
    metadata:dict
    def predict(self,V):return self.feature_map.transform(V)@self.W.T+self.b
    def save_state(self,path,*,Z=None,y=None):
        arrays=dict(components=self.feature_map.components,normalization=self.feature_map.normalization,
          landmark_indices=self.feature_map.original_indices,gamma=self.feature_map.gamma,
          w=self.W,b=self.b,v=self.duals,taus=self.taus,R=self.R,c=self.c,C=self.loss_coefficients)
        if Z is not None:arrays['Z']=Z
        if y is not None:arrays['y']=y
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**arrays)
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'includes_training_arrays':Z is not None and y is not None}

class KernelQualificationError(RuntimeError):
    def __init__(self,message,metadata):super().__init__(message);self.metadata=metadata


def fit_model(method,Xfit,yfit,ns,nt,params,seed,*,taus=TAUS,gap_rtol=1e-5,gap_atol=1e-7,
              max_iter=100,max_seconds_per_head=None,features=None,callback=None):
    start=time.perf_counter();Xfit=np.asarray(Xfit,float);yfit=np.asarray(yfit,float);taus=np.asarray(taus,float)
    if features is None:fm,Z,fi=prepare_features(Xfit,ns,params,seed)
    else:
        fm,Z,fi=features
        assert fi['input_sha256']==digest(Xfit)
    R,c,C,oi=objective(method,Z,yfit,ns,nt,params)
    info={'method':method,'params':dict(params),'seed':int(seed),'features':fi,'objective':oi,
          'tau_grid':taus.tolist(),'label_sha256':digest(yfit),'head_certificates':[],
          'solver':'Structure-exploiting original-dual Mehrotra IPM','objective_changed':False,
          'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).with_name('structured_ipm.py')]}}
    W=[];B=[];duals=[]
    for tau in taus:
        result=solve(Z,yfit,R,c,C,float(tau),gap_rtol=gap_rtol,gap_atol=gap_atol,
             max_iter=max_iter,max_seconds=max_seconds_per_head)
        report=result.report();report['tau']=float(tau);info['head_certificates'].append(report)
        if callback:callback(info)
        if not result.converged:
            info['elapsed_seconds']=time.perf_counter()-start
            raise KernelQualificationError('Original primal/feasible-dual certificate failed',info)
        W.append(result.w);B.append(result.b);duals.append(result.dual_variable)
    info['elapsed_seconds']=time.perf_counter()-start
    return FittedKernel(fm,np.stack(W),np.array(B),np.stack(duals),taus,R,c,C,info)


def fit_predict(method,Xfit,yfit,V,ns,nt,params,seed,**kwargs):
    """Return standardized raw quantiles, JSON-ready diagnostics, fitted state."""
    model=fit_model(method,Xfit,yfit,ns,nt,params,seed,**kwargs)
    started=time.perf_counter();prediction=model.predict(V)
    model.metadata['prediction_seconds']=time.perf_counter()-started
    return prediction,model.metadata,model


def load_model(path,metadata=None):
    with np.load(path,allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files}
    fm=FeatureMap(a['components'],a['normalization'],float(a['gamma']),a['landmark_indices'],{})
    return FittedKernel(fm,a['w'],a['b'],a['v'],a['taus'],a['R'],a['c'],a['C'],metadata or {})


def core_backend(method,task,X,y,V,weight,params,seed,ns,taus=None,final=False):
    """Compatibility adapter for the historical core/qualified baseline wrapper.

    The supplied outside weights are checked against the declared domain mass,
    rather than silently ignored. The same original gap tolerance is used for
    candidate and final fitting; 'final' does not change the objective.
    """
    nt=len(y)-ns
    _,_,expected,_=objective(method,np.zeros((len(y),1)),np.asarray(y,float),ns,nt,params)
    actual=float(params['C'])*np.asarray(weight,float)
    if not np.allclose(actual,expected,rtol=1e-12,atol=1e-12):
        raise ValueError('Outside sample weights differ from declared source/target loss mass')
    levels=[.5] if task=='point' else (TAUS if taus is None else taus)
    prediction,meta,_=fit_predict(method,X,y,V,ns,nt,params,seed,taus=levels)
    meta['certificates']=meta['head_certificates']
    return (prediction[:,0] if task=='point' else prediction),meta
