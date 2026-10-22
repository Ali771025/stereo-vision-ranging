"""Defaults do not contain fabricated camera calibrations or latency values."""
import math


DEFAULTS = dict(file='data/ptz_calibration.json', pairs_dir='data/ptz_pairs',
    timing_file='data/latency_profile.json', stationary_board_confirmed=False,require_foreground_mask=False,
    min_poses=25, min_pan_span_deg=8., min_tilt_span_deg=6., max_fit_reprojection_px=1.,
    settle_s=3., pose_poll_s=.02, http_timeout_s=.5, max_pose_gap_ms=250.,
    max_pose_rtt_ms=100., max_exposure_skew_ms=15., target_speed_bound_m_s=3.,
    timing_min_samples=30, latency_margin_ms=1.)


def configure(cfg):
    given = cfg.get('dynamic', {})
    if not isinstance(given, dict):
        raise ValueError('dynamic must be a mapping')
    result = DEFAULTS | given
    for key, default in DEFAULTS.items():
        value = result[key]
        if isinstance(default, bool):
            if type(value) is not bool:
                raise ValueError('dynamic.'+key+' must be true/false')
        elif isinstance(default, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('dynamic.'+key+' must be finite and positive')
            if isinstance(default, int) and type(value) is not int:
                raise ValueError('dynamic.'+key+' must be an integer')
        elif not isinstance(value, str) or not value:
            raise ValueError('dynamic.'+key+' must be a path')
    if result['min_poses'] < 20 or result['timing_min_samples'] < 10:
        raise ValueError('Need min_poses >= 20 and timing_min_samples >= 10')
    cfg['dynamic'] = result
    return result
