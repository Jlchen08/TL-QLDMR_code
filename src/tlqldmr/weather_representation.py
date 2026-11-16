"""Deterministic, covariate-only representations of 24 past weather observations.

Inputs are already imputed/scaled by the owning chronological fitting fold.
No labels, domain labels, calibration data, or assessment data are accepted.
Fit once on the joint source/target fitting matrix; reuse the same object for
all fitting/validation/inference matrices. This module never changes a model
objective, loss, solver, landmarks, or a bandwidth selection policy.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np

VERSION = 'past_weather_covariate_representation_v2_20261113_2'
MODES = ('history', 'recent', 'mixed', 'last')
STEPS, CHANNELS, INPUT_DIMENSION = 24, 15, 360
SUMMARY_BLOCKS = ('last', 'mean_last4', 'mean_last12', 'mean_all24',
                  'last_minus_first', 'std_all24_ddof0')
CHANNEL_NAMES = (
    'wind_speed_hub', 'wind_speed_50m', 'wind_speed_30m', 'wind_speed_10m',
    'air_temperature', 'atmospheric_pressure', 'relative_humidity',
    'direction_hub_sin', 'direction_hub_cos',
    'direction_50m_sin', 'direction_50m_cos',
    'direction_30m_sin', 'direction_30m_cos',
    'direction_10m_sin', 'direction_10m_cos',
)
DIRECTION_PAIRS = ((7, 8), (9, 10), (11, 12), (13, 14))
SCALE_FLOOR = 1e-12


def _digest(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.dtype.str, array.shape)).encode() + array.tobytes()).hexdigest()


def _matrix(X, *, fitting=False):
    result = np.asarray(X, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != INPUT_DIMENSION:
        raise ValueError('Expected X shape (n,360): 24 chronological steps x15 encoded channels')
    if fitting and not len(result):
        raise ValueError('Cannot fit a representation on zero rows')
    if not np.isfinite(result).all():
        raise ValueError('Use the owning fold fitting-only imputer before this representation')
    return result


def _recent(X):
    """Six 15-channel summaries, entirely inside each strict-past window."""
    if not len(X):
        return np.empty((0, CHANNELS * len(SUMMARY_BLOCKS)), dtype=np.float64)
    windows = X.reshape(-1, STEPS, CHANNELS)
    return np.ascontiguousarray(np.concatenate((
        windows[:, -1, :], windows[:, -4:, :].mean(axis=1),
        windows[:, -12:, :].mean(axis=1), windows.mean(axis=1),
        windows[:, -1, :] - windows[:, 0, :], windows.std(axis=1, ddof=0),
    ), axis=1))


@dataclass(frozen=True)
class WeatherRepresentation:
    """Fitted fold-local feature transform, with immutable covariate statistics."""
    mode: str
    center: np.ndarray
    scale: np.ndarray
    fit_rows: int
    fit_input_sha256: str
    constant_summary_coordinates: tuple

    @property
    def output_dimension(self):
        return {'history': 360, 'recent': 90, 'mixed': 450, 'last': 15}[self.mode]

    @property
    def metadata(self):
        return self.describe()

    def transform(self, X):
        X = _matrix(X)
        if self.mode == 'history':
            # Exact original standardized vector, with no new fitting transform.
            return np.array(X, dtype=np.float64, order='C', copy=True)
        if self.mode == 'last':
            # Same fold-local standardization as history; explicit one-step
            # context control, without refitting or angle conversion.
            return np.array(X[:, -CHANNELS:], dtype=np.float64, order='C', copy=True)
        recent = (_recent(X) - self.center) / self.scale
        if self.mode == 'recent':
            return np.ascontiguousarray(recent)
        # Equal total Euclidean weight for history and summary blocks, rather
        # than allowing 360 history coordinates to outweigh90 summaries.
        return np.ascontiguousarray(np.concatenate((
            X / np.sqrt(2. * INPUT_DIMENSION),
            recent / np.sqrt(2. * CHANNELS * len(SUMMARY_BLOCKS)),
        ), axis=1))

    def describe(self):
        return dict(
            version=VERSION, mode=self.mode, input_dimension=INPUT_DIMENSION,
            output_dimension=self.output_dimension, fit_rows=self.fit_rows,
            fit_input_sha256=self.fit_input_sha256,
            covariates_only=True, labels_accepted=False,
            fit_population='One joint matrix of source and target fitting inputs only; row/domain weights are not inferred here.',
            input_preprocessing='Already imputed and standardized by the owning chronological fitting fold; the360-vector is time-major oldest-to-latest.',
            channel_names=list(CHANNEL_NAMES), direction_pairs=[list(x) for x in DIRECTION_PAIRS],
            summary_blocks=list(SUMMARY_BLOCKS) if self.mode in ('recent', 'mixed') else [],
            summary_normalization='Per-coordinate joint-fit centering; joint-fit population SD. Each sine/cosine pair within a summary block shares its RMS SD. Constant scales become1.' if self.mode in ('recent', 'mixed') else 'Identity on supplied standardized history or last lag; no new normalization.',
            center_sha256=_digest(self.center), scale_sha256=_digest(self.scale),
            constant_summary_coordinates=list(self.constant_summary_coordinates),
            mixed_block_weights={'history': float(1/np.sqrt(720.)), 'recent': float(1/np.sqrt(180.))} if self.mode == 'mixed' else None,
            directions='Encoded sine/cosine pairs retained together; never inverted to wrapped angles or direction degrees.',
            kernel='RBF on this fixed finite-dimensional Euclidean transform is positive semidefinite for every gamma>=0.',
            inference='transform has no fitting side effect; future rows and batch composition cannot change its statistics.',
        )


def fit_representation(X, mode='history'):
    """Fit on joint source+target fitting covariates X only; no y argument."""
    if mode not in MODES:
        raise ValueError('Unknown representation mode: '+str(mode))
    X = _matrix(X, fitting=True)
    if mode in ('history', 'last'):
        center, scale, constant = np.empty(0), np.empty(0), ()
    else:
        summaries = _recent(X)
        center = summaries.mean(axis=0)
        scale = summaries.std(axis=0, ddof=0)
        constant = tuple(np.flatnonzero(scale <= SCALE_FLOOR).tolist())
        for block in range(len(SUMMARY_BLOCKS)):
            for sine, cosine in DIRECTION_PAIRS:
                indices = [block*CHANNELS+sine, block*CHANNELS+cosine]
                pair_scale = np.sqrt(np.mean(scale[indices]**2))
                scale[indices] = pair_scale
        scale = np.where(scale > SCALE_FLOOR, scale, 1.)
        center = np.ascontiguousarray(center)
        scale = np.ascontiguousarray(scale)
    # Arrays are private and read-only: caller mutation cannot alter inference.
    center.setflags(write=False);scale.setflags(write=False)
    return WeatherRepresentation(mode, center, scale, len(X), _digest(X), constant)
