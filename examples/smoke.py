"""Synthetic installation/solver check; this is not a paper experiment."""
from pathlib import Path
import json
import tempfile
import numpy as np
from tlqldmr import kernel_model as K
from tlqldmr.weather_representation import fit_representation
from tlqldmr import metrics

rng=np.random.default_rng(20261116)
X=rng.normal(size=(64,360));y=.7*X[:,-15]-.3*X[:,-14]+rng.normal(scale=.15,size=64)
V=rng.normal(size=(12,360));ns,nt=48,16
rep=fit_representation(X,'last');Z=rep.transform(X);VV=rep.transform(V)
p=dict(C=.5,target_fraction=.5,eta1=.002,eta2=.001,rank=16,
       landmark_strategy='balanced_domains',gamma=.05,regularization_parameterization='effective')
fm,Phi,_=K.prepare_features(Z,ns,p,0);R,c,C,info=K.objective('TL-QLDMR',Phi,y,ns,nt,p)
# Source-centered variance, including the label-dependent linear term.
w=rng.normal(size=Phi.shape[1]);b=.3;u=y[:ns]-Phi[:ns]@w-b
l1,l2=info['lambda1'],info['lambda2'];delta=Phi[:ns].mean(0)-Phi[ns:].mean(0)
left=.5*w@R@w-c@w+.5*l1*np.var(y[:ns])
right=.5*w@w+.5*l1*np.var(u)+.5*l2*(delta@w)**2
assert np.allclose(left,right,rtol=1e-11,atol=1e-11)
assert np.allclose([C[:ns].sum(),C[ns:].sum()],[.5*p['C']*len(y)]*2)
model=K.fit_model('TL-QLDMR',Z,y,ns,nt,p,0,max_iter=100)
raw=model.predict(VV);assert raw.shape==(12,9) and np.isfinite(raw).all()
assert all(q['converged'] and q['gap']<=q['gap_atol']+q['gap_rtol']*max(1,abs(q['primal']),abs(q['dual_bound'])) for q in model.metadata['head_certificates'])
with tempfile.TemporaryDirectory() as td:
 path=Path(td)/'model.npz';model.save_state(path)
 assert np.allclose(K.load_model(path).predict(VV),raw,rtol=1e-13,atol=1e-13)
assert metrics.distribution(rng.normal(size=12),raw,10)['n']==12
print(json.dumps(dict(status='passed',data='synthetic only',native_heads=9,
 original_objective_identity=True,dual_gap_certificates=True,model_roundtrip=True)))
