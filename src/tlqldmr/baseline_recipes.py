"""Predeclared V2 baseline procedures and fit-only instantiation.

This module never accepts evaluation labels. HHO's callback is owned by the
development runner; its 96 calls include initialization and rapid dives.
"""
from __future__ import annotations
import hashlib,json,math
import numpy as np
from scipy.stats import qmc

MODES=['history','recent','mixed','last']
NINE=['HistQGBR','AMQRNN','CQR-MLP','RLMKL','NFS-SVQR','UQSVM-SSVQR']
PAIR=['TSVQR','NuSVR-CI','HybridDL-Interval','QGBR']
POINT=['Standard-SVR','Standard-LDMR','BLSSVR','HHO-SVR','FLSVR','ARA-SVR','KMeans-GBT','RF-WPF','BRF-WPF']
METHODS=NINE+PAIR+POINT
HISTORY_ONLY={'AMQRNN','HybridDL-Interval','NFS-SVQR','ARA-SVR'}
GPU_METHODS={'AMQRNN','HybridDL-Interval','CQR-MLP'}
RELATIVE_KERNEL={'Standard-SVR','Standard-LDMR','BLSSVR','FLSVR','ARA-SVR','TSVQR','UQSVM-SSVQR','NFS-SVQR','NuSVR-CI','RLMKL','HHO-SVR'}
KEYS={
 'Standard-SVR':['C','epsilon'], 'HHO-SVR':['C','epsilon'],
 'Standard-LDMR':['C','rank','epsilon','ldmr_k'], 'BLSSVR':['C','rank','loss_scale'],
 'FLSVR':['C','rank','epsilon'], 'ARA-SVR':['C','epsilon','corr_threshold'],
 'TSVQR':['C','rank','epsilon'], 'UQSVM-SSVQR':['C','rank','epsilon','truncation'],
 'NFS-SVQR':['C','rank','epsilon','selected_features'], 'NuSVR-CI':['C','nu'],
 'RLMKL':['C','rank','scad_lambda','kernel_l1','epochs','lr'],
 'RF-WPF':['trees','depth','leaf','max_features'],
 'QGBR':['trees','depth','leaf','max_features','tree_lr'],
 'HistQGBR':['trees','depth','leaf','max_features','tree_lr','max_leaf_nodes'],
 'KMeans-GBT':['trees','depth','leaf','clusters','tree_lr'],
 'BRF-WPF':['trees','depth','leaf','max_features','width'],
 'AMQRNN':['width','epochs','lr'], 'HybridDL-Interval':['width','epochs','lr','coverage_penalty'],
 'CQR-MLP':['width','epochs','lr','depth'],
}


def task(method):return 'native' if method in NINE else 'pair' if method in PAIR else 'point'
def objectives(method):return ['point','prob95','distribution'] if method in NINE else ['prob95'] if method in PAIR else ['point']
def level(a,values):return values[min(len(values)-1,int(float(a)*len(values)))]


def effective_identity(method,recipe):
    """Ignore inert dictionary fields and known tree-cap equivalences."""
    r=dict(recipe);r.pop('index',None);r.pop('candidate',None);r.pop('v2_recipe',None)
    if method=='QGBR' and r.get('depth') is None:r['depth']=6
    if method=='HistQGBR' and r.get('depth') is not None:
        r['max_leaf_nodes']=min(r['max_leaf_nodes'],2**r['depth'])
    return json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False)


def candidates(method,budget=96):
    assert method in METHODS and method!='HHO-SVR' and budget==96
    seed=20261114+int(hashlib.sha256(method.encode()).hexdigest()[:6],16)
    u=qmc.LatinHypercube(22,seed=seed).random(4096)
    modes=['history'] if method in HISTORY_ONLY else MODES
    quota=budget//len(modes);count={mode:0 for mode in modes};seen=set();rows=[]
    for i,a in enumerate(u):
        mode=modes[i%len(modes)]
        if count[mode]>=quota:continue
        allpar=dict(C=float(10**(-2+4*a[1])),epsilon=float(10**(-3+2*a[3])),
            rank=level(a[4],[128,256,512]),trees=level(a[5],[100,300]),
            depth=level(a[6],[3,6,None]),leaf=level(a[7],[5,15,30]),
            max_features=level(a[8],[.5,1.]),width=level(a[9],[32,64]),
            epochs=level(a[10],[60,120]),lr=level(a[11],[.0003,.0005,.001,.003]),
            clusters=level(a[12],[2,4,6]),nu=float(.1+.8*a[13]),
            tree_lr=level(a[14],[.03,.05,.1]),max_leaf_nodes=level(a[15],[15,31,63]),
            loss_scale=float(10**(-1+2*a[16])),truncation=float(10**(-.6+1.2*a[17])),
            ldmr_k=float(10**(-2+3*a[18])),corr_threshold=level(a[19],[.1,.25,.4,.6]),
            selected_features=level(a[20],[45,90,120,180,360]),
            scad_lambda=float(10**(-4+2*a[16])),kernel_l1=float(10**(-4+2*a[17])),
            coverage_penalty=level(a[21],[.1,1.,10.]))
        if method=='CQR-MLP':allpar['depth']=level(a[6],[1,2,3])
        row=dict(representation=mode,target_fraction=level(a[0],['empirical',.1,.25,.5,.75,.9,.99]),
            **{k:allpar[k] for k in KEYS[method]})
        if method in RELATIVE_KERNEL:row.update(gamma_mode='relative',gamma_multiplier=float(10**(-1.3+2.6*a[2])))
        # The original adapter anchor is counted once in the original-history
        # representation; remaining configurations retain independent strata.
        if not rows:
            row.update(target_fraction=.99)
            anchors=dict(C=50.,epsilon=.01,rank=256,trees=100,depth=3,leaf=5,max_features=.5,
                width=32,epochs=60,lr=.001,tree_lr=.05,nu=.5,ldmr_k=1.,loss_scale=1.,
                truncation=2.,corr_threshold=.4,selected_features=120,scad_lambda=.001,
                kernel_l1=.001,coverage_penalty=1.,max_leaf_nodes=31,clusters=4)
            row.update({k:anchors[k] for k in KEYS[method]})
            if method in RELATIVE_KERNEL:
                row.pop('gamma_multiplier');row.update(gamma_mode='absolute',gamma=.002)
        key=effective_identity(method,row)
        if key in seen:continue
        row.update(index=len(rows),candidate=len(rows),v2_recipe=True)
        rows.append(row);seen.add(key);count[mode]+=1
        if len(rows)==budget:break
    assert len(rows)==budget and all(v==quota for v in count.values()),(method,len(rows),count)
    assert len({effective_identity(method,r) for r in rows})==budget
    return rows


def instantiate(method,recipe,d,V,*,representation_module,kernel_module):
    """Use the current fold's fitting X for every transform and gamma scale."""
    assert recipe.get('v2_recipe') and method in METHODS
    mode=recipe['representation'];assert mode in MODES
    if method in HISTORY_ONLY:assert mode=='history'
    rep=representation_module.fit_representation(d['Xfit'],mode)
    X=rep.transform(d['Xfit']);VV=rep.transform(V)
    ns=int(d['n_source']);nt=int(d['n_target'])
    par={k:recipe[k] for k in KEYS[method]}
    p=recipe['target_fraction'];par['target_fraction']=nt/(ns+nt) if p=='empirical' else float(p)
    dm=None
    if method in RELATIVE_KERNEL:
        if recipe['gamma_mode']=='relative':
            dm=float(kernel_module.fit_distance_scale(X)['median_squared_distance'])
            par['gamma']=float(recipe['gamma_multiplier'])/dm
        elif recipe['gamma_mode']=='absolute':par['gamma']=float(recipe['gamma'])
        else:raise ValueError('Unknown gamma procedure')
    par.update(candidate=int(recipe['index']),allow_task_adaptations=True)
    return X,VV,par,dict(representation=rep.describe(),fit_distance_median=dm,
        relative_gamma_instantiated_from_current_fold_only=True,empirical_mass_instantiated_from_current_fold_only=True)


def hho_rule():
    return dict(algorithm='Harris hawks with four hawks, exploration/soft/hard besiege and Levy rapid dives',
        query_budget=96,population=4,seed=20261114,representation_modes=MODES,
        bounds=dict(log10_C=[-2,2],log10_gamma_multiplier=[-1.3,1.3],log10_epsilon=[-3,-1],mass_coordinate=[0,1],representation_coordinate=[0,1]),
        target_fractions=['empirical',.1,.25,.5,.75,.9,.99],
        anchor=dict(C=50.,gamma_multiplier=1.,epsilon=.01,target_fraction=.99,representation='history'),
        anchor_note='Relative gamma1 replaces the historical absolute gamma anchor for this adaptive cross-fold procedure; static SVR retains the original absolute .002 anchor.',
        failures='All fourfold attempts recorded; failed procedure returns infinity and still consumes one callback. Rapid dives and repeated clipped procedures consume budget.',
        uniqueness='Report actual unique effective procedures; do not claim all96adaptive proposals are unique.')


def hho_search(evaluate,*,budget=96,seed=20261114):
    assert budget==96
    rng=np.random.default_rng(seed);population=4;rule=hho_rule();fractions=rule['target_fractions']
    bounds=np.array([[-2.,2.],[-1.3,1.3],[-3.,-1.],[0.,1.],[0.,1.]])
    positions=rng.uniform(bounds[:,0],bounds[:,1],size=(population,5))
    positions[0]=[np.log10(50),0.,np.log10(.01),.999,0.]
    best=np.inf;rabbit=None;history=[];scores=np.full(population,np.inf)
    def call(position):
        nonlocal best,rabbit
        if len(history)>=budget:return np.inf
        x=np.clip(position,bounds[:,0],bounds[:,1]);index=len(history)
        recipe=dict(v2_recipe=True,index=index,candidate=index,C=float(10**x[0]),
            gamma_mode='relative',gamma_multiplier=float(10**x[1]),epsilon=float(10**x[2]),
            target_fraction=fractions[min(len(fractions)-1,int(x[3]*len(fractions)))],
            representation=MODES[min(3,int(x[4]*4))])
        # Runner failures are persisted at fold level. Infrastructure errors
        # propagate so replay resumes the same adaptive sequence without refund.
        loss=float(evaluate(recipe,index));loss=loss if np.isfinite(loss) else np.inf
        history.append(dict(index=index,recipe=recipe,loss=None if not np.isfinite(loss) else loss))
        if loss<best:best=loss;rabbit=x.copy()
        return loss
    for j in range(population):scores[j]=call(positions[j])
    def levy():
        beta=1.5;s=(math.gamma(1+beta)*np.sin(np.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        return rng.normal(0,s,5)/np.maximum(abs(rng.normal(size=5)),1e-12)**(1/beta)
    while len(history)<budget:
        mean=positions.mean(0);stage=len(history)/budget
        for j in range(population):
            if len(history)>=budget:break
            energy=2*rng.uniform(-1,1)*(1-stage);jump=2*(1-rng.random());r=rng.random()
            prey=positions[0] if rabbit is None else rabbit
            if abs(energy)>=1:
                if rng.random()>=.5:
                    peer=positions[int(rng.integers(population))];proposal=peer-rng.random()*abs(peer-2*rng.random()*positions[j])
                else:proposal=(prey-mean)-rng.random()*(bounds[:,0]+rng.random()*(bounds[:,1]-bounds[:,0]))
                positions[j]=np.clip(proposal,bounds[:,0],bounds[:,1]);scores[j]=call(positions[j])
            elif r>=.5:
                proposal=(prey-positions[j])-energy*abs(jump*prey-positions[j]) if abs(energy)>=.5 else prey-energy*abs(prey-positions[j])
                positions[j]=np.clip(proposal,bounds[:,0],bounds[:,1]);scores[j]=call(positions[j])
            else:
                trial=prey-energy*abs(jump*prey-(positions[j] if abs(energy)>=.5 else mean));first=call(trial)
                if first<scores[j]:positions[j]=np.clip(trial,bounds[:,0],bounds[:,1]);scores[j]=first
                elif len(history)<budget:
                    trial=trial+.01*levy();second=call(trial)
                    if second<scores[j]:positions[j]=np.clip(trial,bounds[:,0],bounds[:,1]);scores[j]=second
    return dict(query_count=len(history),unique_effective_procedures=len({effective_identity('HHO-SVR',r['recipe']) for r in history}),
        best_loss=None if not np.isfinite(best) else best,history=history)
