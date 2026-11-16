"""Structure-exploiting primal-dual IPM for the unchanged pinball objective.

P=.5*w'R*w-c'w + sum C_i rho_tau(y_i-Z_i w-b), R positive definite.
The dual maximizes y'v-.5*(c+Z'v)'R^-1*(c+Z'v), with
C(tau-1)<=v<=C*tau and sum(v)=0.  A Mehrotra predictor/corrector
step exploits the low-rank Hessian ZR^-1 Z' and diagonal box curvature.
Only m-by-m SPD systems are factored. Qualification ALWAYS recomputes the
original primal and a numerically feasible original dual bound.
No jitter or ridge is added to the objective, intercept, or slack variables.
"""
from __future__ import annotations
from dataclasses import dataclass,field
import time
import numpy as np
from scipy.linalg import cho_factor,cho_solve


def weighted_quantile(x,C,tau):
    use=C>0;x=x[use];w=C[use];order=np.argsort(x,kind='stable')
    k=min(np.searchsorted(np.cumsum(w[order]),tau*w.sum(),side='left'),len(order)-1)
    return float(x[order[k]])

def project_dual(v,C,tau):
    lo=(tau-1)*C;hi=tau*C
    left=float(np.min(v-hi));right=float(np.max(v-lo))
    for _ in range(80):
        mid=left+.5*(right-left);q=np.clip(v-mid,lo,hi)
        if q.sum()>0:left=mid
        else:right=mid
        if right-left<=4*np.finfo(float).eps*max(1.,abs(left),abs(right)):break
    q=np.clip(v-(left+.5*(right-left)),lo,hi)
    for _ in range(4):
        s=float(q.sum())
        if not s:break
        room=q-lo if s>0 else hi-q
        for i in np.flatnonzero(room>0)[np.argsort(room[room>0])[::-1]]:
            d=min(abs(s),float(room[i]));q[i]-=np.copysign(d,s);s-=np.copysign(d,s)
            if abs(s)<=np.finfo(float).eps:break
    return q

@dataclass
class Result:
    w:np.ndarray;b:float;dual_variable:np.ndarray
    primal:float;dual_bound:float;gap:float;relative_gap:float
    converged:bool;iterations:int;elapsed_seconds:float;stop_reason:str
    dual_sum_residual:float;dual_box_violation:float
    gap_atol:float;gap_rtol:float
    history:list=field(default_factory=list)
    def report(self):
        return {k:v for k,v in vars(self).items() if k not in {'w','b','dual_variable'}}


def solve(Z,y,R,c,C,tau,*,gap_rtol=1e-5,gap_atol=1e-7,max_iter=100,
          max_seconds=None,check_every=1,callback=None):
    start=time.perf_counter();Z=np.asarray(Z,dtype=float);y=np.asarray(y,dtype=float)
    R=np.asarray(R,dtype=float);c=np.asarray(c,dtype=float);C=np.asarray(C,dtype=float)
    assert Z.ndim==2 and len(y)==len(Z)==len(C) and R.shape==(Z.shape[1],Z.shape[1]) and c.shape==(Z.shape[1],)
    assert np.all(C>=0) and np.any(C>0) and 0<tau<1
    assert all(np.isfinite(a).all() for a in [Z,y,R,c,C])
    assert np.allclose(R,R.T,rtol=1e-12,atol=1e-12)
    active=C>0;Za=Z[active];ya=y[active];Ca=C[active];n,m=Za.shape
    rf=cho_factor(R,lower=True,check_finite=False)
    lo=(tau-1)*Ca;hi=tau*Ca;v=np.zeros(n)
    # Strictly interior boxes and positive complementary multipliers.
    sl=v-lo;su=hi-v;aa=np.ones(n);dd=np.ones(n)
    w=cho_solve(rf,c,check_finite=False);b=weighted_quantile(ya-Za@w,Ca,tau)
    bestP=np.inf;bestD=-np.inf;bestw=w.copy();bestb=b;bestv=v.copy();history=[]
    converged=False;reason='max_iter';numerical_fallbacks=0

    def certificate(it):
        nonlocal bestP,bestD,bestw,bestb,bestv
        cv=project_dual(v,Ca,tau)
        cw=cho_solve(rf,c+Za.T@cv,check_finite=False)
        cb=weighted_quantile(ya-Za@cw,Ca,tau)
        e=ya-Za@cw-cb
        primal=float(.5*cw@R@cw-c@cw+Ca@np.maximum(tau*e,(tau-1)*e))
        h=c+Za.T@cv
        dual=float(ya@cv-.5*h@cho_solve(rf,h,check_finite=False))
        if primal<bestP:bestP=primal;bestw=cw.copy();bestb=cb
        if dual>bestD:bestD=dual;bestv=cv.copy()
        scale=max(1.,abs(bestP),abs(bestD));rawgap=bestP-bestD
        if rawgap < -1e-9*scale:raise FloatingPointError('Original dual exceeds original primal beyond roundoff')
        gap=max(0.,rawgap)
        row={'iteration':it,'primal':bestP,'dual_bound':bestD,'gap':gap,'relative_gap':gap/scale,
             'raw_gap':rawgap,'complementarity':float((aa@sl+dd@su)/(2*n)),
             'dual_sum_residual':float(abs(bestv.sum())),
             'dual_box_violation':float(max(0.,np.max(lo-bestv),np.max(bestv-hi))),
             'stationarity_inf':float(np.max(abs(-ya+Za@w-aa+dd+b))),
             'elapsed_seconds':time.perf_counter()-start,'linear_solve_spectral_fallbacks':numerical_fallbacks}
        history.append(row)
        if callback:callback(dict(row))
        return gap<=gap_atol+gap_rtol*scale

    def max_step(x,dx,fraction):
        bad=dx<0
        return min(1.,fraction*float(np.min(-x[bad]/dx[bad]))) if np.any(bad) else 1.

    converged=certificate(0)
    for it in range(1,max_iter+1):
        if converged:reason='original_duality_gap_verified';break
        w=cho_solve(rf,c+Za.T@v,check_finite=False)
        rd=-ya+Za@w-aa+dd+b;req=float(v.sum())
        mu=float((aa@sl+dd@su)/(2*n))
        diagonal=aa/sl+dd/su
        dinv=1/diagonal;total=float(dinv.sum());zbar=(dinv@Za)/total
        centered=Za-zbar
        M=R+centered.T@(dinv[:,None]*centered)
        try:
            factor=cho_factor(M,lower=True,check_finite=False)
            def linear(rhs):return cho_solve(factor,rhs,check_finite=False)
        except np.linalg.LinAlgError:
            # Regularize Newton direction only, never objective/certificate.
            evals,evecs=np.linalg.eigh((M+M.T)*.5)
            floor=max(np.max(evals)*1e-14,np.finfo(float).tiny)
            inverse=1/np.maximum(evals,floor);numerical_fallbacks+=1
            def linear(rhs):return evecs@(inverse*(evecs.T@rhs))
        def direction(rcl,rcu):
            rhs=-rd-rcl/sl+rcu/su
            dw=linear(centered.T@(dinv*rhs)-zbar*req)
            db=float((dinv@rhs+req)/total-zbar@dw)
            dv=dinv*(rhs-Za@dw-db)
            da=(-rcl-aa*dv)/sl
            de=(-rcu+dd*dv)/su
            return dv,da,de,db
        av,al,au,ab=direction(aa*sl,dd*su)
        ap=min(max_step(sl,av,1.),max_step(su,-av,1.))
        ad=min(max_step(aa,al,1.),max_step(dd,au,1.))
        muaff=float(((aa+ad*al)@(sl+ap*av)+(dd+ad*au)@(su-ap*av))/(2*n))
        sigma=float(np.clip((muaff/max(mu,np.finfo(float).tiny))**3,0,1))
        dv,da,de,db=direction(aa*sl+av*al-sigma*mu,dd*su-av*au-sigma*mu)
        ap=min(max_step(sl,dv,.995),max_step(su,-dv,.995))
        ad=min(max_step(aa,da,.995),max_step(dd,de,.995))
        v+=ap*dv;aa+=ad*da;dd+=ad*de;b+=ad*db
        sl=v-lo;su=hi-v
        if min(sl.min(),su.min(),aa.min(),dd.min())<=0 or not all(np.isfinite(x).all() for x in [v,aa,dd]):
            reason='numerical_interior_failure';break
        w=cho_solve(rf,c+Za.T@v,check_finite=False)
        if it%check_every==0 or it==max_iter:
            converged=certificate(it)
            if converged:reason='original_duality_gap_verified';break
        if max_seconds and time.perf_counter()-start>=max_seconds:reason='time_limit';break
    else:it=max_iter
    if not history or history[-1]['iteration']!=it:converged=certificate(it)
    if converged:reason='original_duality_gap_verified'
    last=history[-1];fullv=np.zeros(len(C));fullv[active]=bestv
    return Result(w=bestw,b=bestb,dual_variable=fullv,primal=bestP,dual_bound=bestD,
        gap=last['gap'],relative_gap=last['relative_gap'],converged=bool(converged),iterations=it,
        elapsed_seconds=time.perf_counter()-start,stop_reason=reason,
        dual_sum_residual=last['dual_sum_residual'],dual_box_violation=last['dual_box_violation'],
        gap_atol=gap_atol,gap_rtol=gap_rtol,history=history)
