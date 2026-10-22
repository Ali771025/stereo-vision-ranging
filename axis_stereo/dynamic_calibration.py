"""Calibrate fixed-lens PTZ axes against one stationary metric chessboard."""
import json
import time
from contextlib import ExitStack

import cv2
import numpy as np

from .axis import AxisHTTP
from .calibration import board, corners
from .config import fingerprint, local
from .dynamic import DynamicGeometry, pose, wrap
from .geometry import StereoGeometry
from .stream import StereoSource
from .timing import TimingProfile


def board_pose(objects, pixels, k, d):
    ok, rv, tv = cv2.solvePnP(objects, pixels, k, d, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError('Board pose could not be recovered')
    r = cv2.Rodrigues(rv)[0]
    if np.min(objects @ r.T + tv.reshape(3), axis=0)[2] <= 0:
        raise ValueError('Board pose behind camera')
    projected = cv2.projectPoints(objects, rv, tv, k, d)[0].reshape(-1, 2)
    error = float(np.sqrt(np.mean(np.sum((projected-pixels.reshape(-1, 2))**2, axis=1))))
    return r.T, -r.T @ tv.reshape(3), error


def least_squares(residual, initial, max_iterations=120):
    """Small damped finite-difference solver, requiring only NumPy/OpenCV."""
    p = np.asarray(initial, float).copy()
    damping = 1e-3
    for _ in range(max_iterations):
        r = residual(p)
        jac = np.column_stack([(residual(p+np.eye(len(p))[i]*1e-6)-r)/1e-6 for i in range(len(p))])
        normal = jac.T @ jac
        delta = np.linalg.solve(normal+damping*np.diag(np.maximum(np.diag(normal), 1e-8)), -jac.T @ r)
        candidate = p+delta
        if np.linalg.norm(residual(candidate)) < np.linalg.norm(r):
            p = candidate
            damping = max(1e-9, damping/3)
            if np.linalg.norm(delta) < 1e-8:
                break
        else:
            damping = min(1e12, damping*10)
    r = residual(p)
    jac = np.column_stack([(residual(p+np.eye(len(p))[i]*1e-6)-r)/1e-6 for i in range(len(p))])
    return p, jac


def fit_camera(observations, k, d, objects, options):
    if len(observations) < options['min_poses']:
        raise ValueError('Insufficient PTZ observations')
    for row in observations:
        pixels = np.asarray(row['pixels'],float)
        if pixels.shape != (len(objects),2) or not np.isfinite(pixels).all():
            raise ValueError('Invalid PTZ chessboard observations')
        if not all(key in row['position'] and np.isfinite(row['position'][key])
                   for key in ('pan','tilt','zoom','focus')):
            raise ValueError('Invalid PTZ calibration encoder/lens values')
    measured = [board_pose(objects, np.asarray(row['pixels'], np.float32).reshape(-1, 1, 2), k, d)
                for row in observations]
    if any(row[2] > options['max_fit_reprojection_px'] for row in measured):
        raise ValueError('Inaccurate per-image board pose; improve board size/focus/corner quality')
    pans = np.asarray([row['position']['pan'] for row in observations])
    reference = [float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(pans))),
                                            np.mean(np.cos(np.radians(pans)))))),
                 float(np.median([row['position']['tilt'] for row in observations]))]
    angles = np.array([[float(wrap(row['position']['pan']-reference[0])),
                        row['position']['tilt']-reference[1]] for row in observations])
    if np.ptp(angles[:, 0]) < options['min_pan_span_deg'] or np.ptp(angles[:, 1]) < options['min_tilt_span_deg']:
        raise ValueError('PTZ calibration needs independent pan AND tilt variation')
    if np.linalg.matrix_rank(angles-angles.mean(axis=0), tol=.1) < 2:
        raise ValueError('Pan/tilt poses lie on one line; capture a two-dimensional grid')
    order = np.random.default_rng(20261019).permutation(len(observations)).tolist()
    count = max(3,len(observations)//5)
    heldout,train = order[:count],order[count:]
    if len(heldout) < 3:
        raise ValueError('Insufficient held-out PTZ poses')
    near = min(train,key=lambda i:float(np.sum(angles[i]**2)))
    initial = np.zeros(13)
    initial[:3] = cv2.Rodrigues(measured[near][0])[0].reshape(3)
    initial[3:6] = measured[near][1]
    candidates = []
    for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        def residual(p, indices=train):
            result = []
            for i in indices:
                row = observations[i]['position']
                r, c = pose(p, row['pan'], row['tilt'], reference, signs)
                result.extend(cv2.Rodrigues(measured[i][0].T @ r)[0].reshape(3))
                result.extend(c-measured[i][1])
            return np.asarray(result)
        parameters, jac = least_squares(residual, initial)
        candidates.append((float(np.linalg.norm(residual(parameters))), parameters, jac, signs))
    _, parameters, jac, signs = min(candidates, key=lambda item: item[0])
    singular = np.linalg.svd(jac, compute_uv=False)
    if singular[-1] <= 1e-9 or singular[0]/singular[-1] > 1e9:
        raise ValueError('PTZ fit is underconstrained; collect a wider independent angle grid')
    errors, angular, centres = [], [], []
    for i in heldout:
        pos = observations[i]['position']
        r, c = pose(parameters, pos['pan'], pos['tilt'], reference, signs)
        prediction = cv2.projectPoints(objects, cv2.Rodrigues(r.T)[0], -r.T @ c, k, d)[0].reshape(-1, 2)
        errors.extend(np.linalg.norm(prediction-np.asarray(observations[i]['pixels']).reshape(-1, 2), axis=1))
        angular.append(float(np.degrees(np.linalg.norm(cv2.Rodrigues(measured[i][0].T @ r)[0]))))
        centres.append(float(np.linalg.norm(c-measured[i][1])))
    p95 = float(np.percentile(errors, 95))
    if not np.isfinite(p95) or p95 > options['max_fit_reprojection_px']:
        raise ValueError('PTZ held-out reprojection error too large; inspect axis model/backlash/calibration data')
    # Keep the model fitted on TRAINING poses: held-out evidence remains independent.
    model = dict(K=k.tolist(), D=d.reshape(-1).tolist(), parameters=parameters.tolist(), reference=reference,
                 signs=list(signs), lens={key: observations[0]['position'][key] for key in ('zoom', 'focus')},
                 pan_domain=[float(np.min(angles[train, 0])), float(np.max(angles[train, 0]))],
                 tilt_domain=[float(np.min(angles[train, 1])), float(np.max(angles[train, 1]))],
                 angle_error_deg=max(angular), centre_error_m=max(centres),
                 heldout_p95_reprojection_px=p95, jacobian_condition=float(singular[0]/singular[-1]))
    for row in observations:
        if any(abs(row['position'][key]-model['lens'][key]) > 1 for key in ('zoom', 'focus')):
            raise ValueError('Lens changed during PTZ calibration')
    return model


def solve(cfg):
    options = cfg['dynamic']
    manifest = json.loads((local(cfg, options['pairs_dir'])/'manifest.json').read_text(encoding='utf-8'))
    if manifest['fingerprint'] != fingerprint(cfg):
        raise ValueError('PTZ observation identity differs from configuration')
    if len(manifest['pairs']) < options['min_poses']:
        raise ValueError('Need more diverse stationary-board PTZ observations')
    _, objects = board(cfg)
    cameras = []
    for i, side in enumerate(('left', 'right')):
        observations = [row[side] for row in manifest['pairs']]
        cameras.append(fit_camera(observations, np.asarray(manifest['intrinsics'][i]['K'], float),
                                  np.asarray(manifest['intrinsics'][i]['D'], float), objects, options))
    data = dict(schema=1, validated=True, fingerprint=fingerprint(cfg), size=manifest['size'], cameras=cameras,
                quality='heldout_board_geometry_only_not_field_distance_acceptance')
    geometry = DynamicGeometry(data)
    references = [dict(pan=c['reference'][0], tilt=c['reference'][1], **c['lens']) for c in cameras]
    baseline = float(np.linalg.norm(geometry.camera_pose(0, references[0])[1]-geometry.camera_pose(1, references[1])[1]))
    measured = cfg['calibration'].get('measured_baseline_m')
    if measured is not None and abs(baseline-measured)/measured > cfg['calibration']['baseline_tolerance_fraction']:
        raise ValueError('PTZ optical-centre baseline disagrees with independent measurement')
    data['reference_baseline_m'] = baseline
    path = local(cfg, options['file'])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temp.replace(path)
    print('Accepted dynamic calibration:', path)
    print('Reference baseline (m):', baseline)


def capture(cfg):
    options = cfg['dynamic']
    if cfg['geometry'].get('normal_view_confirmed') is not True or options['stationary_board_confirmed'] is not True:
        raise ValueError('Confirm Normal view and dynamic.stationary_board_confirmed; the board MUST NOT MOVE')
    geometry = StereoGeometry.load(cfg)
    timing = TimingProfile(cfg)
    shape, objects = board(cfg)
    folder = local(cfg, options['pairs_dir'])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'manifest.json'
    meta = dict(fingerprint=fingerprint(cfg), size=list(geometry.size), pairs=[], timing_profile_hash=timing.file_hash,
                intrinsics=[dict(K=k.tolist(), D=d.tolist()) for k, d in
                            ((geometry.K1, geometry.D1), (geometry.K2, geometry.D2))])
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        if (existing['fingerprint'] != meta['fingerprint'] or existing['intrinsics'] != meta['intrinsics'] or
                existing.get('timing_profile_hash') != timing.file_hash):
            raise ValueError('PTZ capture folder belongs to another rig/intrinsics calibration')
        meta = existing
    with ExitStack() as cleanup:
        clients = []
        for side in ('left', 'right'):
            client = AxisHTTP(cfg['cameras'][side], options['http_timeout_s'])
            cleanup.callback(client.close)
            clients.append(client)
        source = cleanup.enter_context(StereoSource(cfg))
        cleanup.callback(cv2.destroyAllWindows)
        stable_since, last = time.monotonic(), None
        print('Keep board FIXED. Change ONLY camera pan/tilt manually; hold still, SPACE saves, Q exits.')
        while True:
            positions = [client.position() for client in clients]
            if any(position is None for position in positions):
                raise ValueError('PTZ position readback unavailable')
            if any(client.ptz_query('mode').get('mode', '').lower() != 'normal' for client in clients):
                raise ValueError('Calibration requires Normal mode')
            for i, pos in enumerate(positions):
                ref = geometry.meta['ptz_reference'][i]
                if ref is None or any(abs(pos[key]-ref[key]) > 1 for key in ('zoom', 'focus')):
                    raise ValueError('Need calibrated fixed zoom/focus with readable reference')
            if last is None or any(abs(float(wrap(p['pan']-old['pan']))) > .02 or abs(p['tilt']-old['tilt']) > .02
                                   for p, old in zip(positions, last)):
                stable_since = time.monotonic()
                last = positions
            try:
                frames = source.pair()
            except TimeoutError:
                stable_since = time.monotonic()
                if cv2.waitKey(1)&255 == ord('q'):
                    return
                continue
            images = [frame.image for frame in frames]
            if any((im.shape[1], im.shape[0]) != geometry.size for im in images):
                raise ValueError('PTZ capture image dimensions changed')
            points = [corners(image, shape) for image in images]
            views = [image.copy() for image in images]
            for view, pixels in zip(views, points):
                if pixels is not None:
                    cv2.drawChessboardCorners(view, shape, pixels, True)
            canvas = np.hstack(views)
            cv2.putText(canvas, f'FIXED board | poses={len(meta["pairs"])} | SPACE save | Q quit', (10, 25), 0, .6, (0,255,0), 2)
            cv2.imshow('PTZ axis calibration', canvas)
            key = cv2.waitKey(1)&255
            if key == ord('q'):
                return
            if key != ord(' '):
                continue
            exposure_starts = [timing.interval(side,frame)[0] for side,frame in zip(('left','right'),frames)]
            if any(p is None for p in points) or min(exposure_starts)-stable_since < options['settle_s']:
                print('Rejected: hold camera poses stationary longer, board fully visible in BOTH views')
                continue
            after = [client.position() for client in clients]
            if any(p is None for p in after) or any(any(abs(float(wrap(p[k]-q[k]))) > .02 for k in ('pan','tilt'))
                                                   or any(abs(p[k]-q[k]) > 1 for k in ('zoom','focus'))
                                                   for p,q in zip(after, positions)):
                stable_since = time.monotonic()
                print('Rejected: PTZ moved during capture')
                continue
            row = {}
            tag = str(time.time_ns())
            for i, side in enumerate(('left','right')):
                board_pose(objects, points[i], np.asarray(meta['intrinsics'][i]['K']), np.asarray(meta['intrinsics'][i]['D']))
                filename = f'{tag}_{side}.png'
                if not cv2.imwrite(str(folder/filename), images[i]):
                    raise OSError('Cannot save PTZ calibration image')
                row[side] = dict(position=positions[i], pixels=points[i].reshape(-1,2).tolist(), image=filename)
            meta['pairs'].append(row)
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(meta, indent=2), encoding='utf-8')
            temp.replace(path)
            print('Saved PTZ observation:',len(meta['pairs']))
