import json
import time
from pathlib import Path
import cv2
import numpy as np
from .config import local, fingerprint, require_geometry
from .stream import StereoSource, GeometryGuard
from .geometry import StereoGeometry


def board(cfg):
    c = cfg['calibration']
    square = c.get('square_size_m')
    if square is None or not 0 < float(square) < 1:
        raise ValueError('Set square_size_m to measured printed square size, in metres')
    shape = (int(c['board_cols']),int(c['board_rows']))
    # Odd by even inner-corner count helps avoid 180-degree board ambiguity.
    if shape[0] % 2 == shape[1] % 2:
        raise ValueError('Use an asymmetric corner-count pattern, e.g. 9 x 6')
    obj = np.zeros((shape[0]*shape[1],3),np.float32)
    obj[:,:2] = np.mgrid[0:shape[0],0:shape[1]].T.reshape(-1,2)*float(square)
    return shape,obj


def corners(image, shape):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ok, p = cv2.findChessboardCornersSB(gray,shape,cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        p = np.asarray(p,np.float32).reshape(-1,1,2)
    # Force a consistent grid direction; normal upright rig, modest board rotations.
    if ok and (p[-1,0,1]-p[0,0,1]) < 0:
        p = p[::-1].copy()
    return p if ok else None


def capture(cfg):
    require_geometry(cfg, static=True)
    shape,_ = board(cfg)
    dest = local(cfg,cfg['calibration']['pairs_dir'])
    dest.mkdir(parents=True,exist_ok=True)
    manifest = dest/'manifest.json'
    guard = GeometryGuard(cfg)
    try:
        guard.check(force=True)
        meta = {'fingerprint':fingerprint(cfg),'ptz_reference':guard.reference,'pairs':[]}
        if manifest.exists():
            meta = json.loads(manifest.read_text(encoding='utf-8'))
            if meta['fingerprint'] != fingerprint(cfg):
                raise ValueError('Pairs directory belongs to another rig; use a new directory')
            guard.reference = meta['ptz_reference']
            guard.check(force=True)
        print('Hold board STILL. SPACE saves a valid pair; Q exits. Need 20+ varied poses.')
        with StereoSource(cfg) as source:
            while True:
                try:
                    a,b = source.pair()
                except TimeoutError as exc:
                    print(str(exc))
                    if cv2.waitKey(10)&255 == ord('q'):
                        break
                    continue
                guard.check()
                images = [a.image.copy(),b.image.copy()]
                if a.image.shape != b.image.shape:
                    raise ValueError('Camera image dimensions must match')
                points = [corners(im,shape) for im in images]
                for im,p in zip(images,points):
                    if p is not None:
                        cv2.drawChessboardCorners(im,shape,p,True)
                preview = np.hstack(images)
                cv2.putText(preview,f'Pairs {len(meta["pairs"])} | SPACE save | Q exit',(15,30),0,0.8,(0,255,0),2)
                cv2.imshow('Calibration capture',preview)
                key = cv2.waitKey(1)&255
                if key == ord('q'):
                    break
                if key == 32:
                    if any(p is None for p in points):
                        print('Rejected: board must be fully visible in BOTH views')
                        continue
                    guard.check(force=True)
                    tag = f'{time.time_ns()}'
                    names = [f'{tag}_left.png',f'{tag}_right.png']
                    for name,im in zip(names,(a.image,b.image)):
                        if not cv2.imwrite(str(dest/name),im):
                            raise OSError('Cannot save calibration image')
                    meta['pairs'].append({'left':names[0],'right':names[1],
                                          'arrival_skew_ms':abs(a.arrival-b.arrival)*1000})
                    temp = manifest.with_suffix('.tmp')
                    temp.write_text(json.dumps(meta,indent=2),encoding='utf-8')
                    temp.replace(manifest)
                    print(f'Saved {len(meta["pairs"])}')
    finally:
        guard.close()
        cv2.destroyAllWindows()


def fit_stereo(objects, left, right, size):
    rms1,K1,D1,_,_ = cv2.calibrateCamera(objects,left,size,None,None)
    rms2,K2,D2,_,_ = cv2.calibrateCamera(objects,right,size,None,None)
    rms,K1,D1,K2,D2,R,T,E,F = cv2.stereoCalibrate(objects,left,right,K1,D1,K2,D2,size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,150,1e-7))
    R1,R2,P1,P2,Q,roi1,roi2 = cv2.stereoRectify(K1,D1,K2,D2,size,R,T,
                                              flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
    data = dict(K1=K1,D1=D1,K2=K2,D2=D2,R=R,T=T,E=E,F=F,R1=R1,R2=R2,P1=P1,P2=P2,
                Q=Q,size=size,roi1=roi1,roi2=roi2)
    return data,{'mono_left_rms_px':float(rms1),'mono_right_rms_px':float(rms2),'stereo_rms_px':float(rms)}


def epipolar_error(data,left,right):
    values = []
    for a,b in zip(left,right):
        x = cv2.undistortPoints(a,data['K1'],data['D1'],R=data['R1'],P=data['P1']).reshape(-1,2)
        y = cv2.undistortPoints(b,data['K2'],data['D2'],R=data['R2'],P=data['P2']).reshape(-1,2)
        values.extend(np.abs(x[:,1]-y[:,1]).tolist())
    return float(np.percentile(values,95))


def solve(cfg):
    require_geometry(cfg)
    shape,obj = board(cfg)
    c = cfg['calibration']
    dest = local(cfg,c['pairs_dir'])
    meta = json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
    if meta['fingerprint'] != fingerprint(cfg):
        raise ValueError('Capture manifest differs from configuration')
    observations = []
    size = None
    rejected = []
    for pair in meta['pairs']:
        images = [cv2.imread(str(dest/pair[s])) for s in ('left','right')]
        if any(im is None for im in images):
            raise ValueError('Missing calibration pair image')
        if images[0].shape != images[1].shape:
            raise ValueError('Mismatched calibration image sizes')
        this_size = (images[0].shape[1],images[0].shape[0])
        if size is not None and size != this_size:
            raise ValueError('Mixed image sizes in calibration data')
        size = this_size
        pts = [corners(im,shape) for im in images]
        if any(p is None for p in pts):
            rejected.append(pair['left'])
        else:
            observations.append(pts)
    if len(observations) < c['min_pairs']:
        raise ValueError(f'Need at least {c["min_pairs"]} usable pairs; found {len(observations)}')
    # Hold out every fifth pose; never fit the acceptance model on held-out data.
    train = [p for i,p in enumerate(observations) if i%5 != 0]
    validation = [p for i,p in enumerate(observations) if i%5 == 0]
    for camera in (0,1):
        centres = np.array([p[camera].reshape(-1,2).mean(axis=0) for p in train])
        if np.ptp(centres[:,0]) < size[0]*0.2 or np.ptp(centres[:,1]) < size[1]*0.2:
            raise ValueError('Insufficient board coverage: vary position across width AND height')
    split_data,split_report = fit_stereo([obj]*len(train),[p[0] for p in train],[p[1] for p in train],size)
    heldout_error = epipolar_error(split_data,[p[0] for p in validation],[p[1] for p in validation])
    data,report = fit_stereo([obj]*len(observations),[p[0] for p in observations],
                            [p[1] for p in observations],size)
    # Also ensure a usable horizontal model before writing any accepted file.
    geometry = StereoGeometry(data)
    report.update(validation_p95_epipolar_px=heldout_error, split_fit=split_report,
                  baseline_m=geometry.baseline, usable_pairs=len(observations),rejected_files=rejected)
    good = max(report['mono_left_rms_px'],report['mono_right_rms_px'],report['stereo_rms_px'],
               *split_report.values()) <= c['max_rms_px']
    good &= heldout_error <= c['max_validation_epipolar_px']
    measured = c.get('measured_baseline_m')
    if measured is not None:
        if measured <= 0:
            raise ValueError('measured_baseline_m must be positive')
        report['baseline_relative_error'] = abs(geometry.baseline-measured)/measured
        good &= report['baseline_relative_error'] <= c['baseline_tolerance_fraction']
    report['validated'] = bool(good)
    report['fingerprint'] = fingerprint(cfg)
    report['ptz_reference'] = meta['ptz_reference']
    report['square_size_m'] = c['square_size_m']
    out = local(cfg,c['file'])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix('.report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not good:
        # Retain a prior valid calibration intact; do not overwrite with a bad fit.
        raise ValueError('Calibration rejected. See report; collect better, more varied stationary board poses')
    data['metadata'] = np.array(json.dumps(report))
    temp = out.with_suffix('.tmp.npz')
    np.savez_compressed(temp,**data)
    temp.replace(out)
    sample = observations[0]
    # Export a rectified real pair with epipolar guide lines for human review.
    first = next(p for p in meta['pairs'] if p['left'] not in rejected)
    a,b = geometry.rectify(*[cv2.imread(str(dest/first[s])) for s in ('left','right')])
    canvas = np.hstack([a,b])
    for y in range(0,canvas.shape[0],40):
        cv2.line(canvas,(0,y),(canvas.shape[1]-1,y),(0,255,0),1)
    cv2.imwrite(str(out.with_suffix('.rectified.jpg')),canvas)
    print('Accepted calibration saved:',out)
