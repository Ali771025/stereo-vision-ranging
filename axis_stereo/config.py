import hashlib
import json
import math
import re
from pathlib import Path
import yaml


def load(path='config.yaml'):
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg,dict):
        raise ValueError('Configuration must be a YAML mapping')
    for key in ('cameras', 'stream', 'geometry', 'calibration', 'model', 'measurement', 'output'):
        if key not in cfg:
            raise ValueError(f'Missing config section: {key}')
        if not isinstance(cfg[key],dict):
            raise ValueError('Config section must be a mapping: ' + key)
    def positive(section,key):
        value = cfg[section].get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(section+'.'+key+' must be finite and positive')
    for side in ('left','right'):
        if not isinstance(cfg['cameras'].get(side),dict):
            raise ValueError('Missing camera configuration: ' + side)
        if cfg['cameras'][side]['transport'] not in ('rtsp','mjpeg','snapshot'):
            raise ValueError('transport must be rtsp, mjpeg or snapshot')
    for key in ('fps','open_timeout_ms','read_timeout_ms','max_age_ms','max_arrival_skew_ms'):
        positive('stream',key)
    for key in ('fps','open_timeout_ms','read_timeout_ms'):
        if not isinstance(cfg['stream'][key],int):
            raise ValueError('stream.'+key+' must be an integer')
    positive('stream','reconnect_seconds')
    if cfg['stream'].get('rtsp_transport','tcp') not in ('tcp','udp'):
        raise ValueError('stream.rtsp_transport must be tcp or udp')
    if not re.fullmatch(r'[1-9]\d*x[1-9]\d*',str(cfg['stream'].get('resolution',''))):
        raise ValueError('stream.resolution must be WIDTHxHEIGHT with positive integers')
    for section in ('geometry','output','tracking'):
        values = cfg.get(section,{})
        if not isinstance(values,dict):
            raise ValueError('Config section must be a mapping: ' + section)
        for key,value in values.items():
            if key.endswith('_confirmed') or key in ('allow_missing_ptz_readback','save_video'):
                if not isinstance(value,bool):
                    raise ValueError(section+'.'+key+' must be a YAML boolean (true/false without quotes)')
    for key in ('ptz_poll_seconds','pan_tilt_tolerance_deg','zoom_focus_tolerance_units'):
        positive('geometry',key)
    for key in ('min_feature_points','ratio_test','max_epipolar_px','max_reprojection_px',
                'min_disparity_px','max_disparity_px','min_depth_m','max_depth_m',
                'min_parallax_deg','max_relative_depth_spread','ambiguity_ratio','disparity_sigma_px'):
        positive('measurement',key)
    for low,high in (('min_depth_m','max_depth_m'),('min_disparity_px','max_disparity_px')):
        if cfg['measurement'][low] >= cfg['measurement'][high]:
            raise ValueError('measurement.'+low+' must be less than '+high)
    if cfg['measurement']['ratio_test'] >= 1:
        raise ValueError('measurement.ratio_test must be less than 1')
    if cfg['measurement']['ambiguity_ratio'] <= 1:
        raise ValueError('ambiguity_ratio must be greater than 1')
    if not isinstance(cfg['measurement']['min_feature_points'],int) or cfg['measurement']['min_feature_points'] < 2:
        raise ValueError('measurement.min_feature_points must be an integer >= 2')
    for key in ('imgsz','confidence'):
        positive('model',key)
    if not isinstance(cfg['model']['imgsz'],int) or cfg['model']['confidence'] > 1:
        raise ValueError('model.imgsz must be an integer and confidence must be <= 1')
    if not re.fullmatch(r'(cpu|(?:cuda:)?\d+)',str(cfg['model'].get('device','')).lower()):
        raise ValueError('model.device must be cpu or one CUDA index such as "0"')
    if cfg['model'].get('task','detect') not in ('detect','segment'):
        raise ValueError('model.task must be detect or segment')
    for key in ('min_pairs','max_rms_px','max_validation_epipolar_px','baseline_tolerance_fraction'):
        positive('calibration',key)
    if not isinstance(cfg['calibration']['min_pairs'],int) or cfg['calibration']['min_pairs'] < 10:
        raise ValueError('calibration.min_pairs must be an integer >= 10')
    cfg['_root'] = str(path.parent)
    from .dynamic_config import configure
    configure(cfg)
    return cfg


def local(cfg, path):
    p = Path(path)
    return p if p.is_absolute() else Path(cfg['_root']) / p


def fingerprint(cfg):
    # Excludes secrets; any stream or rig identity change requires recalibration.
    fields = {'cameras': [{k: v for k, v in cfg['cameras'][s].items()
                          if k not in ('username_env', 'password_env')} for s in ('left', 'right')],
              'resolution': cfg['stream']['resolution'], 'rig_label': cfg['geometry']['rig_label'],
              'board': {k: cfg['calibration'][k] for k in ('board_cols','board_rows','square_size_m')}}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def require_geometry(cfg, static=False):
    g = cfg['geometry']
    for flag in ('normal_view_confirmed', 'fixed_ptz_focus_confirmed'):
        if g.get(flag) is not True:
            raise ValueError(f'Confirm geometry.{flag} after checking the physical rig / Axis UI.')
    if static and g.get('static_scene_confirmed') is not True:
        raise ValueError('Independent IP streams are NOT exposure-synchronised. '
                         'Use stationary objects and confirm geometry.static_scene_confirmed.')
