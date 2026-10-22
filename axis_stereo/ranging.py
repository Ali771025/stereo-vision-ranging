"""Runtime bridge between received frames, timed PTZ poses and dynamic geometry."""
import csv
import json

from .config import local
from .dynamic import DynamicGeometry, surface_measurement
from .timing import PoseSampler, TimingProfile


class DynamicRanger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.geometry = DynamicGeometry.load(cfg)
        self.timing = TimingProfile(cfg)
        self.sampler = PoseSampler(cfg)
        self.last_positions = None
        self.last_world_point = None

    def __enter__(self):
        self.sampler.__enter__()
        return self

    def __exit__(self, *args):
        self.sampler.__exit__(*args)

    def measure(self, frames, images, boxes,foreground_masks=None):
        self.last_positions = self.last_world_point = None
        options = self.cfg['dynamic']
        intervals = [self.timing.interval(side, frame) for side, frame in zip(('left','right'), frames)]
        skew = max(abs(intervals[0][0]-intervals[1][1]), abs(intervals[0][1]-intervals[1][0]))
        if skew > options['max_exposure_skew_ms']/1000:
            return None, 'exposure_interval_skew_exceeds_limit'
        try:
            estimates = [history.estimate(interval, options) for history, interval in zip(self.sampler.histories, intervals)]
            positions = [estimate[0] for estimate in estimates]
            self.last_positions = positions
            if options['require_foreground_mask'] and (foreground_masks is None or any(m is None for m in foreground_masks)):
                return None,'foreground_mask_required_but_model_did_not_provide_it'
            budget = dict(pose_error_deg=sum(estimate[1] for estimate in estimates),
                          motion_error_m=skew*options['target_speed_bound_m_s'])
            result, status = surface_measurement(self.geometry, images, boxes, positions, self.cfg['measurement'], budget,foreground_masks)
        except ValueError as exc:
            return None, str(exc)
        if result:
            self.last_world_point = result['xyz_world_m']
            result.update(exposure_intervals_s=intervals, exposure_skew_bound_ms=skew*1000,
                          interpolated_positions=positions,
                          pose_interpolation_budget_deg=budget['pose_error_deg'],
                          timing_profile_hash=self.timing.file_hash)
        return result, status


def evaluate(cfg, log_path, truth_path):
    """Match field ground truth to exact logged frame IDs; never ignore missing ranges."""
    records = {}
    with open(log_path, encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('event') == 'tracking':
                if row['frame'] in records:
                    raise ValueError('Duplicate frame id in tracking log')
                records[row['frame']] = row
    with open(truth_path, encoding='utf-8-sig', newline='') as stream:
        truths = list(csv.DictReader(stream))
    if not truths:
        raise ValueError('Empty dynamic ground-truth CSV')
    import math
    results = []
    seen = set()
    for row in truths:
        index = int(row['frame'])
        if index in seen:
            raise ValueError('Duplicate ground-truth frame')
        seen.add(index)
        truth = [float(row['true_range_'+side+'_m']) for side in ('left','right')]
        if any(not math.isfinite(v) or v <= 0 for v in truth):
            raise ValueError('Dynamic reference ranges must be finite and positive')
        record = records.get(index, {})
        measurements = record.get('measurements', [])
        item = dict(frame=index, accepted=False, reason=record.get('range_status','missing_frame'))
        if record.get('fresh') is True and len(measurements) == 1:
            errors = [measurements[0]['range_'+side+'_m']-value for side,value in zip(('left','right'),truth)]
            if all(math.isfinite(value) for value in errors):
                item.update(accepted=True, left_error_m=errors[0], right_error_m=errors[1],
                            within_20cm=all(abs(v) <= .20 for v in errors),
                            calibration_hash=measurements[0].get('calibration_hash'))
        results.append(item)
    summary = dict(samples=len(results), accepted=sum(r['accepted'] for r in results), rows=results,
                   all_samples_within_20cm=all(r.get('within_20cm',False) for r in results),
                   scope='supplied field samples only; no automatic certification of untested distances/speeds')
    path = local(cfg,cfg['output']['directory'])/'dynamic_validation.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return summary
