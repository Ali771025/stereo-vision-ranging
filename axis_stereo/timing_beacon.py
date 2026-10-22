"""Optical host-clock beacon for measuring receive delays without new packages.

Requires measured upper bounds for host-display latency and camera exposure.
It cannot measure those hardware bounds itself. Never use guessed zero bounds.
"""
import csv
import math
import time
import uuid

import cv2
import numpy as np

from .config import local
from .stream import StereoSource


class BeaconClock:
    def __init__(self, display_bound_s, exposure_bound_s):
        if any(not math.isfinite(x) or x <= 0 for x in (display_bound_s, exposure_bound_s)):
            raise ValueError('Measured display and exposure upper bounds must be finite and positive')
        self.display_bound = display_bound_s
        self.exposure_bound = exposure_bound_s
        self.prefix = 'AS1:'+uuid.uuid4().hex+':'
        self.records = {}
        self.current = None
        self.counter = 0

    def issue(self, now):
        if self.current:
            self.records[self.current]['retired'] = now
        token = self.prefix+str(self.counter)
        self.counter += 1
        self.records[token] = dict(issued=now, retired=None)
        self.current = token
        # Bounded memory; excessively old decoded tokens cannot be qualified.
        for old in list(self.records):
            if now-self.records[old]['issued'] > 60:
                del self.records[old]
        return token

    def exposure(self, token, arrival):
        record = self.records.get(token)
        if record is None or record['retired'] is None:
            return None
        low = record['issued']-self.exposure_bound
        high = min(arrival, record['retired']+self.display_bound+self.exposure_bound)
        if low > high:
            return None
        return low, high


def collect(cfg, display_lag_ms, exposure_ms, tick_ms=100., samples=30):
    if not hasattr(cv2,'QRCodeEncoder_create'):
        raise ValueError('This OpenCV build lacks QRCodeEncoder; use independently measured timing CSV instead')
    if not math.isfinite(tick_ms) or tick_ms <= 0 or samples < cfg['dynamic']['timing_min_samples']:
        raise ValueError('Positive tick-ms and samples >= dynamic.timing_min_samples required')
    clock = BeaconClock(display_lag_ms/1000,exposure_ms/1000)
    encoder = cv2.QRCodeEncoder_create()
    decoder = cv2.QRCodeDetector()
    folder = local(cfg,cfg['output']['directory'])
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/'timing_observations.csv'
    if path.exists():
        raise ValueError('timing_observations.csv already exists; archive it before a new measurement')
    pending = []
    counts = dict(left=0,right=0)
    last = -float('inf')
    try:
        cv2.namedWindow('Timing beacon | cameras must see THIS display | Q quit',cv2.WINDOW_NORMAL)
        with path.open('w',newline='',encoding='utf-8') as stream, StereoSource(cfg) as source:
            writer = csv.DictWriter(stream,fieldnames=['side','receive_monotonic_s',
                'exposure_start_monotonic_s','exposure_end_monotonic_s','beacon_token'])
            writer.writeheader()
            while min(counts.values()) < samples:
                now = time.monotonic()
                if now-last >= tick_ms/1000:
                    token = clock.issue(now)
                    qr = encoder.encode(token)
                    qr = cv2.copyMakeBorder(qr,4,4,4,4,cv2.BORDER_CONSTANT,value=255)
                    image = cv2.resize(qr,(720,720),interpolation=cv2.INTER_NEAREST)
                    cv2.imshow('Timing beacon | cameras must see THIS display | Q quit',image)
                    last = now
                try:
                    frames = source.pair(timeout=.01)
                except TimeoutError:
                    frames = None
                if frames:
                    for side,frame in zip(('left','right'),frames):
                        if counts[side] >= samples:
                            continue
                        token,_,_ = decoder.detectAndDecode(frame.image)
                        if token in clock.records:
                            pending.append((side,frame.arrival,token))
                waiting = []
                for side,arrival,token in pending:
                    interval = clock.exposure(token,arrival)
                    if interval is None:
                        if token in clock.records:
                            waiting.append((side,arrival,token))
                        continue
                    if counts[side] < samples:
                        writer.writerow(dict(side=side,receive_monotonic_s=arrival,
                            exposure_start_monotonic_s=interval[0],exposure_end_monotonic_s=interval[1],beacon_token=token))
                        counts[side] += 1
                pending = waiting
                stream.flush()
                if cv2.waitKey(1)&255 in (ord('q'),27):
                    break
    finally:
        cv2.destroyAllWindows()
    print('Timing observations:',path,counts)
    print('Use timing-calibrate on this CSV. Bounds remain empirical, not hardware synchronisation.')
