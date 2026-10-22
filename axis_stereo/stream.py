from collections import deque
from dataclasses import dataclass
import threading
import time
import math
import cv2
import numpy as np
from .axis import AxisHTTP, open_rtsp, params, redacted_error


@dataclass
class Frame:
    image: np.ndarray
    arrival: float
    sequence: int
    exposure_interval: tuple | None = None


class Reader:
    """Each reader owns its decoder/session; bounded queue, automatic reconnect."""
    def __init__(self, camera, stream):
        self.camera, self.stream = camera, stream
        self.frames = deque(maxlen=12)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.status = 'starting'
        self.sequence = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _publish(self, image):
        self.sequence += 1
        with self.lock:
            self.frames.append(Frame(image, time.monotonic(), self.sequence))
        self.status = 'OK'

    def _mjpeg(self, client):
        with client.get('/axis-cgi/mjpg/video.cgi', params(self.camera, self.stream), stream=True) as r:
            buf = b''
            for chunk in r.iter_content(4096):
                if self.stop_event.is_set():
                    return
                buf += chunk
                while True:
                    start = buf.find(b'\xff\xd8')
                    end = buf.find(b'\xff\xd9', max(start, 0))
                    if start < 0 or end < 0:
                        break
                    image = cv2.imdecode(np.frombuffer(buf[start:end+2], np.uint8), cv2.IMREAD_COLOR)
                    buf = buf[end+2:]
                    if image is not None:
                        self._publish(image)
                if len(buf) > 16_000_000:
                    raise ValueError('MJPEG buffer overflow')
            raise ConnectionError('MJPEG ended')

    def _run(self):
        while not self.stop_event.is_set():
            cap = None
            client = None
            try:
                mode = self.camera['transport']
                if mode == 'rtsp':
                    cap = open_rtsp(self.camera, self.stream)
                    if not cap.isOpened():
                        raise ConnectionError('RTSP open failed')
                    while not self.stop_event.is_set():
                        ok, image = cap.read()
                        if not ok:
                            raise ConnectionError('RTSP read failed')
                        self._publish(image)
                else:
                    client = AxisHTTP(self.camera, self.stream['read_timeout_ms'] / 1000)
                    if mode == 'mjpeg':
                        self._mjpeg(client)
                    elif mode == 'snapshot':
                        while not self.stop_event.is_set():
                            self._publish(client.snapshot(self.stream))
                            self.stop_event.wait(1 / self.stream['fps'])
                    else:
                        raise ValueError('Unknown transport')
            except Exception as exc:
                self.status = redacted_error(exc) + ': reconnecting'
                with self.lock:
                    self.frames.clear()
                self.stop_event.wait(self.stream['reconnect_seconds'])
            finally:
                if cap is not None:
                    cap.release()
                if client is not None:
                    client.close()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=8)


def choose_pair(left, right, now, max_skew, max_age):
    """Consume queues; never reuse a frame. Times are RECEIVE times, not exposure."""
    while left and now - left[0].arrival > max_age:
        left.popleft()
    while right and now - right[0].arrival > max_age:
        right.popleft()
    if not left or not right:
        return None
    choices = [(abs(a.arrival-b.arrival), i, j) for i, a in enumerate(left) for j, b in enumerate(right)]
    eligible = [choice for choice in choices if choice[0] <= max_skew]
    if not eligible:
        if left[-1].arrival < right[0].arrival:
            left.clear()
        elif right[-1].arrival < left[0].arrival:
            right.clear()
        return None
    # Prefer the newest usable pair, rather than an old perfectly aligned pair
    # accumulated while YOLO was busy. Skew breaks ties in recency.
    skew,i,j = min(eligible,key=lambda c: (-min(left[c[1]].arrival,right[c[2]].arrival),c[0]))
    a, b = left[i], right[j]
    for _ in range(i+1):
        left.popleft()
    for _ in range(j+1):
        right.popleft()
    return a, b


class StereoSource:
    def __init__(self, cfg):
        self.cfg = cfg
        self.left = Reader(cfg['cameras']['left'], cfg['stream'])
        self.right = Reader(cfg['cameras']['right'], cfg['stream'])

    def __enter__(self):
        self.left.start()
        try:
            self.right.start()
        except BaseException:
            self.left.close()
            raise
        return self

    def pair(self, timeout=5):
        end = time.monotonic() + timeout
        s = self.cfg['stream']
        while time.monotonic() < end:
            with self.left.lock, self.right.lock:
                pair = choose_pair(self.left.frames, self.right.frames, time.monotonic(),
                                   s['max_arrival_skew_ms']/1000, s['max_age_ms']/1000)
            if pair:
                return pair
            time.sleep(0.005)
        raise TimeoutError('No fresh pair. left=' + self.left.status + ', right=' + self.right.status)

    def __exit__(self, *_):
        self.left.close()
        self.right.close()


class GeometryGuard:
    def __init__(self, cfg, reference=None):
        self.cfg = cfg
        self.clients = []
        try:
            for side in ('left','right'):
                self.clients.append(AxisHTTP(cfg['cameras'][side]))
        except BaseException:
            self.close()
            raise
        self.reference = reference
        self.last = -float('inf')

    def check(self, force=False):
        g = self.cfg['geometry']
        if not force and time.monotonic()-self.last < g['ptz_poll_seconds']:
            return
        positions = [c.position() for c in self.clients]
        for p in positions + (self.reference or []):
            if p is not None and (not all(k in p for k in ('pan','tilt','zoom','focus')) or
                                  not all(math.isfinite(v) for v in p.values())):
                raise ValueError('Invalid PTZ/focus telemetry')
        if any(p is None for p in positions) and not g['allow_missing_ptz_readback']:
            raise ValueError('PTZ/focus readback unavailable. Check permissions; see manual fallback in README_FA.')
        if self.reference is None:
            self.reference = positions
        for old, new in zip(self.reference, positions):
            if old is None:
                continue  # explicit manual fallback
            if new is None:
                raise ValueError('PTZ telemetry lost since calibration')
            for k in old:
                delta = abs(old[k]-new[k])
                if k == 'pan':
                    delta = min(delta, abs(360-delta))
                tol = g['pan_tilt_tolerance_deg'] if k in ('pan','tilt') else g['zoom_focus_tolerance_units']
                if delta > tol:
                    raise ValueError(f'Geometry changed: {k}. Restore EXACT settings or recalibrate.')
        self.last = time.monotonic()

    def close(self):
        for c in self.clients:
            c.close()
