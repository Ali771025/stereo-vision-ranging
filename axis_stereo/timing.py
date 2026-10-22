"""Empirical receive-delay profiles and time-bracketed PTZ histories.

Profiles describe measured samples; they do not turn IP streams into hardware
exposure synchronisation. Clock values must share the host monotonic timebase.
"""
from collections import deque
import csv
import json
import math
import threading
import time

from .axis import AxisHTTP
from .config import local
from .dynamic import digest, identity, wrap


def calibrate(cfg, csv_path):
    with open(csv_path, encoding='utf-8-sig', newline='') as source:
        rows = list(csv.DictReader(source))
    options = cfg['dynamic']
    report = dict(schema=1, identity=identity(cfg), validated=True, cameras={},
                  interpretation='empirical_latency_interval_not_hardware_sync')
    for side in ('left', 'right'):
        observations = []
        receive_times = set()
        for row in rows:
            if row['side'] != side:
                continue
            arrival, start, end = [float(row[k]) for k in
                                   ('receive_monotonic_s', 'exposure_start_monotonic_s', 'exposure_end_monotonic_s')]
            if not all(math.isfinite(x) for x in (arrival, start, end)) or not start <= end <= arrival:
                raise ValueError('Timing CSV needs finite exposure_start <= exposure_end <= receive in SAME host clock')
            if arrival in receive_times:
                raise ValueError('Duplicate receive timestamp; timing samples must be independent per camera')
            receive_times.add(arrival)
            observations.append((arrival-end, arrival-start))
        if len(observations) < options['timing_min_samples']:
            raise ValueError('Insufficient independent timing samples for ' + side)
        train = [r for i, r in enumerate(observations) if i % 5]
        holdout = [r for i, r in enumerate(observations) if not i % 5]
        margin = options['latency_margin_ms']/1000
        low = max(0., min(r[0] for r in train)-margin)
        high = max(r[1] for r in train)+margin
        if not all(low <= a <= b <= high for a, b in holdout):
            raise ValueError('Held-out latency outside bounds for ' + side + '; collect representative measurements')
        report['cameras'][side] = dict(min_delay_s=low, max_delay_s=high, samples=len(observations))
    report['minimum_possible_skew_budget_ms'] = sum(
        c['max_delay_s']-c['min_delay_s'] for c in report['cameras'].values())*500
    report['potentially_usable_for_dynamic_ranges'] = (
        report['minimum_possible_skew_budget_ms'] <= options['max_exposure_skew_ms'])
    path = local(cfg, options['timing_file'])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temp.replace(path)
    print(json.dumps(report, indent=2))


class TimingProfile:
    def __init__(self, cfg):
        self.data = json.loads(local(cfg, cfg['dynamic']['timing_file']).read_text(encoding='utf-8'))
        self.file_hash = digest(local(cfg,cfg['dynamic']['timing_file']))
        if self.data.get('schema') != 1 or self.data.get('validated') is not True or self.data.get('identity') != identity(cfg):
            raise ValueError('Timing profile invalid or belongs to different decoder/computer/stream settings')
        for side in ('left', 'right'):
            values = self.data['cameras'][side]
            low, high = values['min_delay_s'], values['max_delay_s']
            if not all(math.isfinite(x) for x in (low, high)) or not 0 <= low <= high:
                raise ValueError('Invalid timing interval')

    def interval(self, side, frame):
        # Future adapters can supply independently verified exposure intervals.
        if frame.exposure_interval is not None:
            low, high = frame.exposure_interval
            if not all(math.isfinite(x) for x in (low, high)) or not low <= high <= frame.arrival:
                raise ValueError('Invalid frame exposure interval')
            return low, high
        delay = self.data['cameras'][side]
        return frame.arrival-delay['max_delay_s'], frame.arrival-delay['min_delay_s']


class PoseHistory:
    def __init__(self):
        self.samples = deque(maxlen=1000)
        self.lock = threading.Lock()

    def add(self, start, end, position):
        if not all(math.isfinite(x) for x in (start, end)) or start > end or position is None:
            raise ValueError('Invalid timed PTZ sample')
        if not all(k in position and math.isfinite(position[k]) for k in ('pan', 'tilt', 'zoom', 'focus')):
            raise ValueError('Incomplete/nonfinite timed PTZ position')
        with self.lock:
            if self.samples and (start+end)/2 <= sum(self.samples[-1][:2])/2:
                raise ValueError('PTZ timestamps must increase')
            self.samples.append((start, end, dict(position)))

    def clear(self):
        with self.lock:
            self.samples.clear()

    def estimate(self, interval, options):
        low, high = interval
        with self.lock:
            samples = list(self.samples)
        midpoints = [(a+b)/2 for a, b, _ in samples]
        if not samples or low < midpoints[0] or high > midpoints[-1]:
            raise ValueError('PTZ history does not bracket exposure interval')
        centre = (low+high)/2
        index = next((i for i in range(len(samples)-1) if midpoints[i] <= centre <= midpoints[i+1]), None)
        if index is None:
            raise ValueError('No PTZ interpolation bracket')
        # Every gap crossed by the uncertain exposure must be qualified.
        relevant = [i for i in range(len(samples)-1) if midpoints[i] <= high and midpoints[i+1] >= low]
        for i in relevant:
            if midpoints[i+1]-midpoints[i] > options['max_pose_gap_ms']/1000:
                raise ValueError('PTZ telemetry gap too large')
            if any(samples[j][1]-samples[j][0] > options['max_pose_rtt_ms']/1000 for j in (i, i+1)):
                raise ValueError('PTZ telemetry round trip too large')
        a, b = samples[index][2], samples[index+1][2]
        # Lens changes anywhere in the interval invalidate interpolation.
        for i in relevant:
            if any(abs(samples[i][2][key]-samples[i+1][2][key]) > 1 for key in ('zoom', 'focus')):
                raise ValueError('Lens changed during exposure interval')
        gap = midpoints[index+1]-midpoints[index]
        alpha = (centre-midpoints[index])/gap
        pan_step = float(wrap(b['pan']-a['pan']))
        position = dict(pan=float(wrap(a['pan']+alpha*pan_step)),
                        tilt=a['tilt']+alpha*(b['tilt']-a['tilt']), zoom=a['zoom'], focus=a['focus'])
        speeds, widths = [], []
        for i in relevant:
            duration = midpoints[i+1]-midpoints[i]
            before, after = samples[i][2], samples[i+1][2]
            speeds.append((abs(float(wrap(after['pan']-before['pan'])))+abs(after['tilt']-before['tilt']))/duration)
            widths.append(max(samples[j][1]-samples[j][0] for j in (i, i+1))/2+duration/2)
        # Approximate constant-velocity budget, explicitly not a hard motor bound.
        error = max(speeds)*(max(widths)+(high-low)/2)
        return position, float(error)


class PoseSampler:
    """Each polling thread owns a separate HTTP session; never share control sessions."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.histories = [PoseHistory(), PoseHistory()]
        self.stop_event = threading.Event()
        self.status = ['starting', 'starting']
        self.threads = [threading.Thread(target=self._poll, args=(i, side), daemon=True)
                        for i, side in enumerate(('left', 'right'))]

    def _poll(self, index, side):
        client = None
        try:
            client = AxisHTTP(self.cfg['cameras'][side], self.cfg['dynamic']['http_timeout_s'])
            while not self.stop_event.is_set():
                try:
                    if client.ptz_query('mode').get('mode', '').lower() != 'normal':
                        raise ValueError('Camera not in Normal mode')
                    start = time.monotonic()
                    value = client.position()
                    end = time.monotonic()
                    self.histories[index].add(start, end, value)
                    self.status[index] = 'OK'
                except Exception as exc:
                    self.histories[index].clear()
                    self.status[index] = type(exc).__name__
                self.stop_event.wait(self.cfg['dynamic']['pose_poll_s'])
        finally:
            if client:
                client.close()

    def __enter__(self):
        started = []
        try:
            for thread in self.threads:
                thread.start()
                started.append(thread)
        except BaseException:
            self.stop_event.set()
            for thread in started:
                thread.join(timeout=5)
            raise
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=5)
