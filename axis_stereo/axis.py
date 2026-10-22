import os
import re
import socket
import math
from urllib.parse import quote, urlencode
import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth, HTTPBasicAuth
from .errors import redacted_error


def credentials(camera):
    u = os.environ.get(camera['username_env'])
    p = os.environ.get(camera['password_env'])
    if not u or not p:
        raise ValueError('Set credential environment variables: ' + camera['username_env'] + ', ' + camera['password_env'])
    return u, p


def params(camera, stream, codec=None):
    p = {'camera': camera['channel'], 'resolution': stream['resolution'], 'fps': stream['fps']}
    if codec:
        p.update(videocodec=codec, audio=0)
    return p


def rtsp_url(camera, stream):
    u, p = credentials(camera)
    return (f"rtsp://{quote(u, safe='')}:{quote(p, safe='')}@{camera['host']}:"
            f"{camera['rtsp_port']}/axis-media/media.amp?{urlencode(params(camera, stream, 'h264'))}")


class AxisHTTP:
    def __init__(self, camera, timeout=3):
        self.camera, self.timeout = camera, timeout
        self.session = requests.Session()
        self.session.trust_env = False  # LAN cameras must not pass through system proxies
        u, p = credentials(camera)
        self.digest, self.basic = HTTPDigestAuth(u, p), HTTPBasicAuth(u, p)
        self.base = f"{camera.get('http_scheme', 'http')}://{camera['host']}:{camera['http_port']}"

    def get(self, path, query=None, stream=False):
        args = dict(params=query, timeout=(self.timeout, self.timeout), stream=stream,
                    allow_redirects=False)
        r = self.session.get(self.base + path, auth=self.digest, **args)
        if r.status_code == 401 and 'basic' in r.headers.get('WWW-Authenticate', '').lower():
            r.close()
            r = self.session.get(self.base + path, auth=self.basic, **args)
        try:
            r.raise_for_status()
            if 300 <= r.status_code < 400:
                raise ValueError('Camera redirected HTTP; set http_scheme / http_port explicitly.')
        except Exception:
            r.close()
            raise
        return r

    def snapshot(self, stream):
        with self.get('/axis-cgi/jpg/image.cgi', params(self.camera, stream)) as r:
            frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('HTTP response is not a JPEG frame')
        return frame

    def position(self):
        try:
            with self.get('/axis-cgi/com/ptz.cgi', {'query': 'position', 'camera': self.camera['channel']}) as r:
                found = dict(re.findall(r'(?m)^([a-z]+)=([^\r\n]+)', r.text))
            result = {k: float(found[k]) for k in ('pan', 'tilt', 'zoom', 'focus') if k in found}
            return result if all(k in result and math.isfinite(result[k])
                                 for k in ('pan', 'tilt', 'zoom', 'focus')) else None
        except (requests.RequestException, ValueError):
            return None

    def close(self):
        self.session.close()

    def ptz_query(self, query):
        with self.get('/axis-cgi/com/ptz.cgi',
                      {'query': query, 'camera': self.camera['channel']}) as r:
            if r.text.lstrip().lower().startswith('error'):
                raise ValueError('Camera rejected PTZ query: ' + query)
            return dict(re.findall(r'(?m)^([a-zA-Z]+)=([^\r\n]+)', r.text))

    def ptz_move(self, pan_delta, tilt_delta, speed):
        """Finite relative move; never leave a continuous velocity command active."""
        with self.get('/axis-cgi/com/ptz.cgi', {
            'camera': self.camera['channel'], 'rpan': pan_delta,
            'rtilt': tilt_delta, 'speed': speed}) as r:
            if r.text.lstrip().lower().startswith('error'):
                raise ValueError('Camera rejected relative PTZ movement')

    def ptz_stop(self):
        with self.get('/axis-cgi/com/ptz.cgi',
                      {'camera': self.camera['channel'], 'move': 'stop'}) as r:
            if r.text.lstrip().lower().startswith('error'):
                raise ValueError('Camera rejected PTZ stop')


def open_rtsp(camera, stream):
    # Process-wide options set before creating any FFmpeg capture.
    transport = stream.get('rtsp_transport', 'tcp')
    if transport not in ('tcp', 'udp'):
        raise ValueError('stream.rtsp_transport must be tcp or udp')
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;' + transport
    # FFmpeg can print complete credential URLs on errors: silence its backend log.
    os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
    return cv2.VideoCapture(rtsp_url(camera, stream), cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, stream['open_timeout_ms'],
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, stream['read_timeout_ms']])


def port_open(host, port):
    try:
        with socket.create_connection((host, int(port)), timeout=2):
            return True
    except OSError:
        return False

