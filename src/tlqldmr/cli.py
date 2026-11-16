"""Portable CLI around the frozen scientific kernel; labels are absent at inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from . import kernel_model as K, metrics as M
from .weather_representation import fit_representation, WeatherRepresentation


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def matrix(x, name):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 360 or not np.isfinite(x).all():
        raise ValueError(name + ' must be finite (n,360) standardized past-weather rows')
    return x


def save_predictions(path, raw):
    # Preserve original native outputs. Nominal95 uses only the two specified
    # endpoint heads; distribution metrics independently rearrange all heads.
    raw = np.asarray(raw)
    np.savez_compressed(path, raw=raw, taus=K.TAUS,
                        point=M.integrated_mean(raw),
                        median=np.sort(raw, axis=1)[:, 4],
                        original_tau_0_5=raw[:, 4],
                        nominal95=np.sort(raw[:, [0, -1]], axis=1))


def fit(args):
    config = json.loads(args.config.read_text())
    d = read_npz(args.fit); v = read_npz(args.inference)
    if set(v) != {'X'}:
        raise ValueError('Inference archive must contain X only; no outcome labels')
    X = matrix(d['Xfit'], 'Xfit'); V = matrix(v['X'], 'X')
    y = np.asarray(d['yfit'], dtype=float)
    ns, nt = int(d['n_source']), int(d['n_target'])
    if ns < 0 or nt < 0 or ns+nt != len(X) or y.shape != (len(X),) or not np.isfinite(y).all():
        raise ValueError('Invalid domain counts or standardized fitting labels')
    ym, ys = float(d['label_mean']), float(d['label_scale'])
    if not np.isfinite(ym) or not np.isfinite(ys) or ys <= 0:
        raise ValueError('Finite label mean and positive label scale required')
    p = dict(config['params'])
    rep = fit_representation(X, p['representation'])
    XX, VV = rep.transform(X), rep.transform(V)
    scale = K.fit_distance_scale(XX)
    # Original final runner instantiates the relative bandwidth on fitting
    # covariates only. Recorded absolute gamma remains in the released config.
    p['gamma'] = float(p['gamma_multiplier']) / scale['median_squared_distance']
    p['regularization_parameterization'] = 'effective'
    numeric = dict(config.get('numerical', {}))
    numeric.setdefault('gap_rtol', 1e-5); numeric.setdefault('gap_atol', 1e-7)
    numeric.setdefault('max_iter', 100); numeric.setdefault('max_seconds_per_head', 120)
    args.output.mkdir(parents=True, exist_ok=False)
    method = args.method or config.get('method', 'TL-QLDMR')
    model = K.fit_model(method, XX, y, ns, nt, p, args.seed, taus=K.TAUS, **numeric)
    raw = model.predict(VV)*ys+ym
    if not np.isfinite(raw).all(): raise ArithmeticError('Nonfinite prediction')
    model.save_state(args.output/'model.npz')
    np.savez_compressed(args.output/'representation.npz', center=rep.center,
                        scale=rep.scale, mode=rep.mode, fit_rows=rep.fit_rows,
                        fit_input_sha256=rep.fit_input_sha256,
                        constant_summary_coordinates=np.asarray(rep.constant_summary_coordinates,dtype=np.int64),
                        label_mean=ym, label_scale=ys)
    save_predictions(args.output/'predictions.npz', raw)
    save_json(args.output/'fit_report.json', dict(config_sha256=file_sha(args.config),
        fit_archive_sha256=file_sha(args.fit), inference_archive_sha256=file_sha(args.inference),
        method=method, seed=args.seed, fit_source_rows=ns, fit_target_rows=nt,
        representation=rep.describe(), fit_distance_scale=scale,
        instantiated_params=p, metadata=model.metadata,
        scientific_scope='Fixed-feature fit and prediction; no selection or assessment labels read'))
    print(json.dumps(dict(status='complete', output=str(args.output), heads=len(K.TAUS))))


def predict(args):
    v = read_npz(args.inference)
    if set(v) != {'X'}: raise ValueError('Inference archive must contain X only')
    state = read_npz(args.model/'representation.npz')
    rep = WeatherRepresentation(str(state['mode']), state['center'], state['scale'],
        int(state['fit_rows']), str(state['fit_input_sha256']),
        tuple(state['constant_summary_coordinates'].tolist()))
    model = K.load_model(args.model/'model.npz')
    raw = model.predict(rep.transform(matrix(v['X'], 'X')))*float(state['label_scale'])+float(state['label_mean'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists(): raise FileExistsError(args.output)
    save_predictions(args.output, raw)
    print(json.dumps(dict(status='complete', output=str(args.output))))


def score(args):
    q = read_npz(args.predictions); d = read_npz(args.targets)
    y = np.asarray(d['y'], dtype=float).reshape(-1); cap = float(d['capacity_mw'])
    raw = np.asarray(q['raw'], dtype=float)
    if raw.shape != (len(y), 9) or not np.isfinite(raw).all() or not np.isfinite(y).all() or not cap > 0:
        raise ValueError('Require matching finite predictions/targets and positive capacity')
    if not np.array_equal(q['taus'], K.TAUS): raise ValueError('Expected native nine-head grid')
    result = dict(point=M.point(y,M.integrated_mean(raw),cap),
                  original_tau_0_5=M.point(y,raw[:,4],cap),
                  nominal95=M.interval(y,raw[:,[0,-1]],cap),
                  native_distribution=M.distribution(y,raw,cap))
    save_json(args.output, result)
    print(json.dumps(dict(status='scored', output=str(args.output), rows=len(y))))


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command',required=True)
    f = sub.add_parser('fit',help='Fit standardized source-then-target rows; predict label-free inputs')
    f.add_argument('--fit',type=Path,required=True); f.add_argument('--inference',type=Path,required=True)
    f.add_argument('--config',type=Path,required=True); f.add_argument('--output',type=Path,required=True)
    f.add_argument('--seed',type=int,default=0)
    f.add_argument('--method',choices=['TL-QLDMR','Standard-SVQR','Weighted-SVQR','NoVariance','NoAlignment'])
    f.set_defaults(func=fit)
    q = sub.add_parser('predict',help='Load a saved model and representation; never fit')
    q.add_argument('--model',type=Path,required=True); q.add_argument('--inference',type=Path,required=True)
    q.add_argument('--output',type=Path,required=True); q.set_defaults(func=predict)
    s = sub.add_parser('score',help='Separate assessment of frozen predictions')
    s.add_argument('--predictions',type=Path,required=True); s.add_argument('--targets',type=Path,required=True)
    s.add_argument('--output',type=Path,required=True); s.set_defaults(func=score)
    a=p.parse_args();a.func(a)

if __name__ == '__main__': main()
