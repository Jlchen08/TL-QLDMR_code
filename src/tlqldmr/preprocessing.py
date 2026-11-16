"""Fit-only preprocessing and domain/split helpers extracted from the formal study.

These functions operate on already constructed strict-past weather windows.
See docs/data_format.md for the public input contract and stage boundaries.
"""
from __future__ import annotations
import warnings
import numpy as np

def physical_weather(raw):
    """Recover wind/temperature once per original ID, without future records."""
    input_ids=raw['input_ids'];unique,first=np.unique(input_ids.ravel(),return_index=True)
    encoded=raw['X'].reshape(-1,15)
    n=int(max(input_ids.max(),raw['ids'].max()))+1
    wind=np.full(n,np.nan);temperature=np.full(n,np.nan)
    wind[unique]=encoded[first,0];temperature[unique]=encoded[first,4]
    times=np.full(n,np.datetime64('NaT','ns'),dtype='datetime64[ns]')
    windows,positions=first//24,first%24
    times[unique]=raw['times'][windows]-(24-positions)*np.timedelta64(15,'m')
    wind[raw['ids']]=raw['target_wind'];temperature[raw['ids']]=raw['target_temperature'];times[raw['ids']]=raw['times']
    return wind,temperature,times


def thresholds(weather,fit_start,fit_end):
    wind,temp,times=weather
    historical=(times>=np.datetime64(fit_start))&(times<np.datetime64(fit_end))
    w=wind[historical&np.isfinite(wind)];t=temp[historical&np.isfinite(temp)]
    assert len(w)>0 and len(t)>0
    return dict(wind_ms=float(np.percentile(w,98)),temperature_low_c=float(np.percentile(t,2)),
        temperature_high_c=float(np.percentile(t,98)),cutout_ms=25.,percentile=98,
        fit_start=fit_start,fit_end=fit_end,training_valid_wind_count=len(w),training_valid_temperature_count=len(t),
        threshold_scope='Unique physical observations strictly inside this fold fitting period; no validation weather used')


def extreme(raw,rule):
    return ((raw['target_wind']>rule['wind_ms'])|
        (raw['target_temperature']<rule['temperature_low_c'])|
        (raw['target_temperature']>rule['temperature_high_c'])|(raw['target_wind']>25.))


def time_mask(times,start,end,*,purge_start=True,purge_end=True):
    start=np.datetime64(start);end=np.datetime64(end)
    lower=start+np.timedelta64(6,'h') if purge_start else start
    upper=end-np.timedelta64(6,'h') if purge_end else end
    return (times>=lower)&(times<upper)


def transform(X,mean,scale):return np.ascontiguousarray((np.where(np.isfinite(X),X,mean)-mean)/scale)


def fit_arrays(X,y,ns,ids,input_ids,times,capacity,Xval,yval,val_ids,val_inputs,val_times):
    assert ns>0 and len(y)-ns>0 and len(yval)>0
    assert len(ids)==len(np.unique(ids)) and len(val_ids)==len(np.unique(val_ids))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);mean=np.nanmean(X,axis=0)
    mean=np.where(np.isfinite(mean),mean,0.);filled=np.where(np.isfinite(X),X,mean)
    scale=filled.std(axis=0);scale=np.where(scale>1e-12,scale,1.)
    ym=float(y.mean());ys=float(y.std());assert ys>1e-12
    return dict(Xfit=transform(X,mean,scale),yfit=(y-ym)/ys,Xval=transform(Xval,mean,scale),yval=np.asarray(yval),
        feature_mean=mean,feature_scale=scale,label_mean=np.array(ym),label_scale=np.array(ys),
        n_source=np.array(ns),n_target=np.array(len(y)-ns),fit_ids=ids,fit_input_ids=input_ids,fit_times=times,
        validation_ids=val_ids,validation_input_ids=val_inputs,validation_times=val_times,
        capacity_mw=np.array(capacity),score_capacity=np.array(capacity),label_unit=np.array('MW'))


def audit_arrays(arrays):
    fit=np.unique(np.r_[arrays['fit_ids'],arrays['fit_input_ids'].ravel()])
    val=np.unique(np.r_[arrays['validation_ids'],arrays['validation_input_ids'].ravel()])
    overlap=np.intersect1d(fit,val)
    assert len(overlap)==0,'Training and validation share original measurement IDs'
    assert arrays['fit_times'].max()<arrays['validation_times'].min()-np.timedelta64(6,'h')
    assert np.isfinite(arrays['Xfit']).all() and np.isfinite(arrays['Xval']).all()
    assert np.isfinite(arrays['yfit']).all() and np.isfinite(arrays['yval']).all()
    assert len(arrays['Xfit'])==int(arrays['n_source'])+int(arrays['n_target'])
    return dict(n_source=int(arrays['n_source']),n_target=int(arrays['n_target']),validation=len(arrays['yval']),
        fit_time_first=str(arrays['fit_times'].min()),fit_time_last=str(arrays['fit_times'].max()),
        validation_time_first=str(arrays['validation_times'].min()),validation_time_last=str(arrays['validation_times'].max()),
        shared_raw_measurement_ids=len(overlap),no_sampling_cap=True,
        imputation_scaling_fitting_rows_only=True,fit_and_validation_original_observation_disjoint=True,
        yfit_is_standardized_yval_in_MW=True)

