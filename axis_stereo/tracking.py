"""Single manually selected person per view, bounded independent PTZ control.

Optional ranges use a separate PTZ-axis calibration and empirical timing model.
The static stereo calibration is never used for moving-camera triangulation.
"""
import json
import csv
import math
import time
from contextlib import ExitStack
from datetime import datetime, timezone

import cv2
import numpy as np

from .axis import AxisHTTP, redacted_error
from .config import local
from .runtime import Detector
from .stream import StereoSource


DEFAULTS = dict(person_class_id=0, min_confidence=0.6, min_iou=0.15,
                max_appearance_distance=0.35, ambiguity_margin=0.12,
                deadband=0.04, gain_deg=8.0, max_step_deg=1.5,
                command_interval_s=0.1, speed=30, servo_gain=.8,
                fixed_mounts_confirmed=False, fixed_zoom_focus_confirmed=False)


def settings(cfg):
    s = DEFAULTS | cfg.get('tracking', {})
    for key in ('min_confidence', 'min_iou', 'max_appearance_distance',
                'ambiguity_margin', 'deadband'):
        if isinstance(s[key],bool) or not isinstance(s[key],(int,float)) or not math.isfinite(s[key]) or not 0 < s[key] < 1:
            raise ValueError('tracking.' + key + ' must be between 0 and 1')
    for key in ('gain_deg', 'max_step_deg', 'command_interval_s','servo_gain'):
        if isinstance(s[key],bool) or not isinstance(s[key],(int,float)) or not math.isfinite(s[key]) or s[key] <= 0:
            raise ValueError('tracking.' + key + ' must be finite and positive')
    if type(s['speed']) is not int:
        raise ValueError('tracking.speed must be an integer')
    if s['max_step_deg'] > 2 or not 1 <= s['speed'] <= 30:
        raise ValueError('Tracking requires max_step_deg <= 2 and speed between 1 and 30')
    if s['servo_gain'] > 1:
        raise ValueError('tracking.servo_gain must be <= 1')
    if type(s['person_class_id']) is not int or s['person_class_id'] < 0:
        raise ValueError('tracking.person_class_id must be a nonnegative integer')
    return s


def overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1)*max(0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.0


def appearance(image, box):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    # Use the central torso to reduce background and leg-motion changes.
    xa, xb = max(0, int(x1+.2*(x2-x1))), min(w, int(x2-.2*(x2-x1)))
    ya, yb = max(0, int(y1+.2*(y2-y1))), min(h, int(y1+.7*(y2-y1)))
    if xb <= xa or yb <= ya:
        return None
    hsv = cv2.cvtColor(image[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    return cv2.normalize(hist, None, norm_type=cv2.NORM_L1)


class Target:
    """Conservative appearance + overlap association, not biometric identity.

    Reference appearance is frozen to limit drift. Any failed/ambiguous update
    latches loss; only explicit operator selection can acquire a person again.
    """
    def __init__(self, options):
        self.options = options
        self.clear()

    def clear(self):
        self.box = self.reference = self.mask = None

    def eligible(self, d):
        return d.cls == self.options['person_class_id'] and d.confidence >= self.options['min_confidence']

    def select(self, image, detections, point):
        self.clear()
        x, y = point
        choices = [d for d in detections if self.eligible(d) and
                   d.box[0] <= x <= d.box[2] and d.box[1] <= y <= d.box[3]]
        if len(choices) != 1:
            return False
        d = choices[0]
        ref = appearance(image, d.box)
        if ref is None:
            return False
        self.box, self.reference = d.box, ref
        self.mask = d.mask
        return True

    def update(self, image, detections):
        if self.box is None:
            return None
        candidates = []
        for d in detections:
            iou = overlap(self.box, d.box)
            if not self.eligible(d) or iou < self.options['min_iou']:
                continue
            hist = appearance(image, d.box)
            if hist is None:
                continue
            distance = cv2.compareHist(self.reference, hist, cv2.HISTCMP_BHATTACHARYYA)
            if distance <= self.options['max_appearance_distance']:
                candidates.append((iou-distance, d))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates or (len(candidates) > 1 and
                candidates[0][0]-candidates[1][0] < self.options['ambiguity_margin']):
            self.clear()
            return None
        self.box = candidates[0][1].box
        self.mask = candidates[0][1].mask
        return self.box


def correction(box, shape, options):
    h, w = shape[:2]
    ex = ((box[0]+box[2])/2-w/2)/(w/2)
    ey = (h/2-(box[1]+box[3])/2)/(h/2)
    def step(error):
        if abs(error) <= options['deadband']:
            return 0.0
        return float(np.clip(error*options['gain_deg'],
                             -options['max_step_deg'], options['max_step_deg']))
    return step(ex), step(ey)


class PTZController:
    def __init__(self, clients, options, enabled):
        self.clients, self.options, self.enabled = clients, options, enabled
        self.last_command = -float('inf')
        self.limits = []
        self.reference = []
        self.geometry = None
        self.frame_positions = None
        self.world_point = None

    def preflight(self):
        for client in self.clients:
            mode = client.ptz_query('mode')
            if mode.get('mode', '').lower() != 'normal':
                raise ValueError('PTZ tracking requires verified Normal mode on both cameras')
            limits = client.ptz_query('limits')
            try:
                bounds = {k: float(limits[k]) for k in ('MinPan', 'MaxPan', 'MinTilt', 'MaxTilt')}
            except (KeyError,TypeError,ValueError) as exc:
                raise ValueError('Camera did not report numeric pan/tilt limits required for tracking') from exc
            if not all(math.isfinite(v) for v in bounds.values()) or any(
                    bounds['Min'+axis] >= bounds['Max'+axis] for axis in ('Pan', 'Tilt')):
                raise ValueError('Invalid camera PTZ limits')
            position = client.position()
            if position is None or not all(math.isfinite(v) for v in position.values()):
                raise ValueError('PTZ tracking requires pan, tilt, zoom and focus telemetry')
            self.limits.append(bounds)
            self.reference.append(position)

    def stop(self):
        errors = []
        if self.enabled:
            for side, client in zip(('left', 'right'), self.clients):
                try:
                    client.ptz_stop()
                except Exception as exc:
                    errors.append(side + ':' + redacted_error(exc))
        if errors:
            raise ValueError('PTZ stop unconfirmed: ' + ', '.join(errors))

    def move(self, boxes, images, arrivals, max_age):
        if time.monotonic()-self.last_command < self.options['command_interval_s']:
            return []
        commands = []
        # Validate BOTH cameras before issuing either command.
        for side, (client, ref, limits, box, image) in enumerate(zip(
                self.clients, self.reference, self.limits, boxes, images)):
            if client.ptz_query('mode').get('mode', '').lower() != 'normal':
                raise ValueError('Camera left Normal mode; select target again after restoring rig')
            pos = client.position()
            if pos is None or not all(math.isfinite(v) for v in pos.values()):
                raise ValueError('Lost PTZ telemetry')
            if any(abs(pos[k]-ref[k]) > 1 for k in ('zoom', 'focus')):
                raise ValueError('Zoom/focus changed during tracking')
            pan, tilt = correction(box, image.shape, self.options)
            if self.geometry is not None:
                if self.frame_positions is None:
                    raise ValueError('Cannot aim calibrated PTZ without image-time pose; wait for qualified timing')
                pixel = ((box[0]+box[2])/2,(box[1]+box[3])/2)
                pan,tilt = self.geometry.aim(side,pos,self.frame_positions[side],pixel,image.shape,
                    self.options['deadband'],self.options['servo_gain'],self.world_point)
                pan = float(np.clip(pan,-self.options['max_step_deg'],self.options['max_step_deg']))
                tilt = float(np.clip(tilt,-self.options['max_step_deg'],self.options['max_step_deg']))
                candidate = dict(pos, pan=pos['pan']+pan, tilt=pos['tilt']+tilt)
                self.geometry.camera_pose(side,candidate)
            if not (limits['MinPan'] <= pos['pan']+pan <= limits['MaxPan'] and
                    limits['MinTilt'] <= pos['tilt']+tilt <= limits['MaxTilt']):
                raise ValueError('Tracking reached PTZ travel limit')
            commands.append((pan, tilt))
        self.last_command = time.monotonic()
        for client, (pan, tilt) in zip(self.clients, commands):
            # HTTP telemetry/control can block. Recheck age before EACH command.
            if time.monotonic()-min(arrivals) > max_age:
                raise ValueError('Frames expired before PTZ command')
            if self.enabled and (pan or tilt):
                client.ptz_move(pan, tilt, self.options['speed'])
        return commands


def run(cfg, enable_ptz=False, max_frames=None, ranges=False):
    options = settings(cfg)
    if max_frames is not None and max_frames <= 0:
        raise ValueError('max_frames must be positive')
    if cfg['geometry'].get('normal_view_confirmed') is not True:
        raise ValueError('Confirm geometry.normal_view_confirmed before tracking')
    if enable_ptz and not all(options[k] is True for k in
                              ('fixed_mounts_confirmed', 'fixed_zoom_focus_confirmed')):
        raise ValueError('Confirm tracking.fixed_mounts_confirmed and fixed_zoom_focus_confirmed')
    detector = Detector(cfg)
    ranger = None
    if ranges:
        from .ranging import DynamicRanger
        ranger = DynamicRanger(cfg)
    targets = [Target(options), Target(options)]
    clients = []
    controller = None
    writer = None
    state = dict(armed=False, images=None, detections=None, pending_clicks=[],
                 message='Click SAME person in BOTH views; SPACE to arm')
    window = 'Single target | SPACE arm/pause | R reset | S snapshot | Q quit'
    out = local(cfg, cfg['output']['directory'])
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')

    def reset(message):
        state.update(armed=False, message=message, pending_clicks=[])
        for target in targets:
            target.clear()
        controller.stop()

    def click(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN or state['images'] is None:
            return
        # OpenCV swallows callback exceptions: do all I/O in the main loop.
        state['pending_clicks'].append((x, y))

    def select_pending():
        if state['images'] is None:
            return
        clicks, state['pending_clicks'] = state['pending_clicks'], []
        for x, y in clicks:
            state['armed'] = False
            controller.stop()
            width = state['images'][0].shape[1]
            side = int(x >= width)
            ok = targets[side].select(state['images'][side], state['detections'][side], (x-side*width, y))
            state['message'] = ('Selected. Click SAME person in other view; SPACE to arm' if ok
                                else 'Selection ambiguous/empty; click one person')

    try:
        for side in ('left', 'right'):
            clients.append(AxisHTTP(cfg['cameras'][side]))
        controller = PTZController(clients, options, enable_ptz)
        controller.geometry = ranger.geometry if ranger else None
        controller.preflight()
        controller.stop()
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, click)
        with (out/f'tracking_{stamp}.jsonl').open('w', encoding='utf-8') as log, \
                StereoSource(cfg) as source, ExitStack() as cleanup:
            if ranger:
                cleanup.enter_context(ranger)
            # Stop motors BEFORE waiting for capture workers to finish closing.
            cleanup.callback(controller.stop)
            csv_file = cleanup.enter_context((out/f'tracking_{stamp}.csv').open('w',newline='',encoding='utf-8-sig'))
            fields = ['utc','frame','range_left_m','range_right_m','support','parallax_deg',
                      'approximate_error_budget_m','within_20cm_budget','quality']
            csv_writer = csv.DictWriter(csv_file,fieldnames=fields)
            csv_writer.writeheader()
            count = 0
            while max_frames is None or count < max_frames:
                try:
                    a, b = source.pair()
                except TimeoutError:
                    reset('Stream unavailable; reselect BOTH targets')
                    state['images'] = None
                    unavailable = np.zeros((160, 900, 3), np.uint8)
                    cv2.putText(unavailable, state['message'], (10, 60), 0, .6, (0, 180, 255), 2)
                    cv2.imshow(window, unavailable)
                    log.write(json.dumps(dict(event='stream_wait', measurements=[]))+'\n')
                    log.flush()
                    if cv2.waitKey(1)&255 in (ord('q'), 27):
                        break
                    continue
                images = (a.image, b.image)
                if images[0].shape != images[1].shape:
                    raise ValueError('Camera dimensions differ')
                detections = detector.predict(list(images))
                selected = [target.box is not None for target in targets]
                for target, image, dets in zip(targets, images, detections):
                    target.update(image, dets)
                fresh = time.monotonic()-min(a.arrival, b.arrival) <= cfg['stream']['max_age_ms']/1000
                if not fresh or any(was and t.box is None for was, t in zip(selected, targets)):
                    reset('Stale/ambiguous/lost target; reselect BOTH targets')
                measurements = []
                range_status = 'disabled_use_track_ranges'
                if ranger and fresh and all(t.box is not None for t in targets):
                    measurement, range_status = ranger.measure((a,b),images,[t.box for t in targets],
                                                               [t.mask for t in targets])
                    if measurement is not None:
                        measurement.update(class_id=options['person_class_id'],class_name='selected_target')
                        measurements = [measurement]
                    controller.frame_positions = ranger.last_positions
                    controller.world_point = ranger.last_world_point
                elif ranger:
                    range_status = 'select_same_target_in_both_views'
                # Feature processing and HTTP may have consumed the age budget.
                fresh = time.monotonic()-min(a.arrival,b.arrival) <= cfg['stream']['max_age_ms']/1000
                if not fresh:
                    measurements = []
                    range_status = 'stale_range_result'
                    reset('Stale processing; reselect BOTH targets')
                commands = []
                if state['armed'] and ranger and controller.frame_positions is None:
                    state.update(armed=False,message='Timing/pose unavailable; SPACE resumes only after qualification')
                    controller.stop()
                if state['armed']:
                    commands = controller.move([t.box for t in targets], images,
                                                (a.arrival, b.arrival), cfg['stream']['max_age_ms']/1000)
                if time.monotonic()-min(a.arrival,b.arrival) > cfg['stream']['max_age_ms']/1000:
                    measurements = []
                    fresh = False
                    range_status = 'stale_after_control'
                    reset('Stale after control; reselect BOTH targets')
                state.update(images=images if fresh else None, detections=detections)
                views = []
                for side,(image, dets, target) in enumerate(zip(images, detections, targets)):
                    view = image.copy()
                    for d in dets:
                        if target.eligible(d):
                            x1, y1, x2, y2 = map(int, d.box)
                            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 180, 255), 1)
                    if target.box is not None:
                        x1, y1, x2, y2 = map(int, target.box)
                        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = ('PTZ ENABLED' if enable_ptz else 'DRY RUN') + ' | ' + ('ARMED' if state['armed'] else 'PAUSED')
                    if measurements:
                        m = measurements[0]
                        value = m['range_left_m' if side==0 else 'range_right_m']
                        range_text = f'EST {value:.2f}m | budget {m["approximate_error_budget_m"]:.2f}m'
                    else:
                        range_text = 'range N/A'
                    cv2.putText(view, label+' | '+range_text, (10, 25), 0, .55, (0, 255, 255), 2)
                    views.append(view)
                canvas = np.hstack(views)
                cv2.putText(canvas, state['message'], (10, canvas.shape[0]-15), 0, .5, (0, 255, 255), 1)
                cv2.putText(canvas,range_status,(10,50),0,.45,(0,255,255),1)
                if cfg['output']['save_video']:
                    if writer is None:
                        writer = cv2.VideoWriter(str(out/f'tracking_{stamp}.avi'),cv2.VideoWriter_fourcc(*'MJPG'),
                                                 cfg['stream']['fps'],(canvas.shape[1],canvas.shape[0]))
                        if not writer.isOpened():
                            raise OSError('Could not create tracking video output')
                    writer.write(canvas)
                cv2.imshow(window, canvas)
                count += 1
                utc = datetime.now(timezone.utc).isoformat()
                log.write(json.dumps(dict(utc=utc, frame=count,
                    event='tracking', armed=state['armed'], ptz_enabled=enable_ptz, fresh=fresh,
                    boxes=[t.box for t in targets], commands=commands,
                    measurements=measurements, range_status=range_status))+'\n')
                log.flush()
                for measurement in measurements:
                    csv_writer.writerow({key:(dict(utc=utc,frame=count)|measurement).get(key) for key in fields})
                csv_file.flush()
                key = cv2.waitKey(1)&255
                select_pending()
                if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord('r'):
                    reset('Click SAME person in BOTH views; SPACE to arm')
                elif key == ord(' '):
                    if state['armed']:
                        state['armed'] = False
                        controller.stop()
                    elif fresh and all(t.box is not None for t in targets):
                        state.update(armed=True, message='Following selected target | R reset | SPACE pause')
                elif key == ord('s'):
                    cv2.imwrite(str(out/f'tracking_{stamp}_{count}.jpg'),canvas)
    finally:
        try:
            if controller:
                controller.stop()
        finally:
            if writer:
                writer.release()
            for client in clients:
                client.close()
            cv2.destroyAllWindows()
