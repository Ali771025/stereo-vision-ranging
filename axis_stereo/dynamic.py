"""Calibrated independent PTZ kinematics and unrectified metric triangulation.

World coordinates belong to the stationary calibration board. A reported range
is a supported visible-surface estimate, not the person's centre of mass.
"""
import hashlib
import json
import math
import platform

import cv2
import numpy as np

from .config import fingerprint, local


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(cfg):
    value = dict(rig=fingerprint(cfg), stream=cfg['stream'], opencv=cv2.__version__,
                 computer=platform.node())
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def wrap(angle):
    return (np.asarray(angle)+180.) % 360.-180.


def rotation(axis, angle):
    vector = np.zeros(3)
    vector[axis] = angle
    return cv2.Rodrigues(vector)[0]


def pose(parameters, pan, tilt, reference, signs=(1, 1)):
    """13-parameter orthogonal-axis model with optical-centre offsets.

    Gauge fixed: pan-to-tilt offset has zero Y, optical offset has zero X.
    Rotations map camera axes to the board/world frame. Units: metres/radians;
    encoder angles and reference are degrees.
    """
    p = np.asarray(parameters, float)
    if p.shape != (13,) or not np.isfinite(p).all():
        raise ValueError('Invalid PTZ kinematic parameters')
    a, b = np.radians([float(wrap(pan-reference[0])), tilt-reference[1]])*signs
    base, mount = cv2.Rodrigues(p[:3])[0], cv2.Rodrigues(p[8:11])[0]
    rp, rt = rotation(1, a), rotation(0, b)
    centre = p[3:6] + base @ rp @ (np.array([p[6], 0., p[7]]) + rt @ np.array([0., p[11], p[12]]))
    return base @ rp @ rt @ mount, centre


def validate_camera(camera):
    for key, shape in (('K', (3, 3)), ('parameters', (13,)), ('reference', (2,)), ('signs', (2,))):
        value = np.asarray(camera[key], float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid dynamic calibration field: ' + key)
    k = np.asarray(camera['K'], float)
    if min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0., 0., 1.]):
        raise ValueError('Invalid dynamic camera intrinsics')
    d = np.asarray(camera['D'], float)
    if d.size not in (4, 5, 8, 12, 14) or not np.isfinite(d).all():
        raise ValueError('Invalid dynamic distortion')
    if any(s not in (-1, 1) for s in camera['signs']):
        raise ValueError('Invalid PTZ axis signs')
    for key in ('pan_domain', 'tilt_domain'):
        domain = np.asarray(camera[key], float)
        if domain.shape != (2,) or not np.isfinite(domain).all() or domain[0] >= domain[1]:
            raise ValueError('Invalid calibrated angular domain')
    for key in ('angle_error_deg', 'centre_error_m'):
        if not math.isfinite(camera[key]) or camera[key] < 0:
            raise ValueError('Invalid calibrated pose error')
    if not all(k in camera.get('lens',{}) and math.isfinite(camera['lens'][k]) for k in ('zoom','focus')):
        raise ValueError('Invalid calibrated lens state')


class DynamicGeometry:
    def __init__(self, data):
        self.data = data
        if data.get('schema') != 1 or data.get('validated') is not True:
            raise ValueError('Dynamic calibration not validated')
        self.cameras = data['cameras']
        if len(self.cameras) != 2:
            raise ValueError('Dynamic calibration requires both cameras')
        for camera in self.cameras:
            validate_camera(camera)
        self.size = tuple(data['size'])
        if len(self.size) != 2 or any(type(v) is not int or v <= 0 for v in self.size):
            raise ValueError('Invalid dynamic calibration image size')

    @classmethod
    def load(cls, cfg):
        path = local(cfg, cfg['dynamic']['file'])
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['fingerprint'] != fingerprint(cfg):
            raise ValueError('Dynamic calibration belongs to different cameras/streams/rig')
        instance = cls(data)
        instance.file_hash = digest(path)
        return instance

    def camera_pose(self, side, position):
        camera = self.cameras[side]
        if not all(k in position and math.isfinite(position[k]) for k in ('pan', 'tilt', 'zoom', 'focus')):
            raise ValueError('Invalid dynamic PTZ position')
        if any(abs(position[key]-camera['lens'][key]) > 1 for key in ('zoom', 'focus')):
            raise ValueError('Dynamic calibration zoom/focus changed')
        pan = float(wrap(position['pan']-camera['reference'][0]))
        tilt = position['tilt']-camera['reference'][1]
        for value, key in ((pan, 'pan_domain'), (tilt, 'tilt_domain')):
            lo, hi = camera[key]
            if not lo-1e-6 <= value <= hi+1e-6:
                raise ValueError('PTZ outside calibrated angular domain; collect wider calibration')
        return pose(camera['parameters'], position['pan'], position['tilt'], camera['reference'], camera['signs'])

    def triangulate(self, a, b, positions, limits):
        a, b = [np.asarray(x, float).reshape(-1, 2) for x in (a, b)]
        if len(a) != len(b):
            raise ValueError('Dynamic correspondence count mismatch')
        poses = [self.camera_pose(i, position) for i, position in enumerate(positions)]
        if not len(a):
            return np.empty((0, 3)), np.zeros(0, bool), np.zeros(0), np.zeros(0)
        normalized, projections = [], []
        for pixels, camera, (r, centre) in zip((a, b), self.cameras, poses):
            normalized.append(cv2.undistortPoints(pixels.reshape(-1, 1, 2),
                              np.asarray(camera['K']), np.asarray(camera['D'])).reshape(-1, 2))
            projections.append(np.column_stack((r.T, -r.T @ centre)))
        h = cv2.triangulatePoints(*projections, normalized[0].T, normalized[1].T).T
        valid = np.isfinite(h).all(axis=1) & (np.abs(h[:, 3]) > 1e-12)
        xyz = np.full((len(a), 3), np.nan)
        xyz[valid] = h[valid, :3]/h[valid, 3, None]
        error = np.zeros(len(a))
        rays = []
        for pixels, camera, (r, centre) in zip((a, b), self.cameras, poses):
            camera_xyz = (xyz-centre) @ r
            valid &= (camera_xyz[:, 2] >= limits['min_depth_m']) & (camera_xyz[:, 2] <= limits['max_depth_m'])
            projected, _ = cv2.projectPoints(np.nan_to_num(xyz), cv2.Rodrigues(r.T)[0], -r.T @ centre,
                                             np.asarray(camera['K']), np.asarray(camera['D']))
            error = np.maximum(error, np.linalg.norm(projected.reshape(-1, 2)-pixels, axis=1))
            ray = xyz-centre
            with np.errstate(invalid='ignore', divide='ignore'):
                rays.append(ray/np.linalg.norm(ray, axis=1)[:, None])
        angle = np.degrees(np.arccos(np.clip(np.sum(rays[0]*rays[1], axis=1), -1., 1.)))
        valid &= np.isfinite(xyz).all(axis=1) & np.isfinite(error) & np.isfinite(angle)
        valid &= (error <= limits['max_reprojection_px']) & (angle >= limits['min_parallax_deg'])
        return xyz, valid, error, angle

    def aim(self, side, current, exposure_position, pixel, shape, deadband, gain=.8, world_point=None):
        """Image-time bearing -> current actuator correction, including mount roll.

        A measured 3D surface point also compensates optical-centre translation.
        Otherwise only the world bearing is held; no fictitious depth is used.
        """
        camera = self.cameras[side]
        k,d = np.asarray(camera['K']),np.asarray(camera['D'])
        ray = cv2.undistortPoints(np.asarray(pixel,float).reshape(1,1,2),k,d).reshape(2)
        image_ray = np.r_[ray,1.]
        r,_ = self.camera_pose(side,exposure_position)
        direction = r @ image_ray
        self.camera_pose(side,current) # Validate current lens/domain before linearisation.
        h,w = shape[:2]
        centre_ray = cv2.undistortPoints(np.array([[[w/2,h/2]]],float),k,d).reshape(2)
        def residual(position):
            rot,c = pose(camera['parameters'],position['pan'],position['tilt'],camera['reference'],camera['signs'])
            vector = direction if world_point is None else np.asarray(world_point,float)-c
            cam = rot.T @ vector
            if cam[2] <= 0:
                raise ValueError('Selected target bearing is behind current camera')
            return cam[:2]/cam[2]-centre_ray
        error = residual(current)
        if abs(error[0]*k[0,0]/(w/2)) <= deadband:
            error[0] = 0.
        if abs(error[1]*k[1,1]/(h/2)) <= deadband:
            error[1] = 0.
        base = residual(current)
        jac = np.column_stack([(residual(dict(current,**{axis:current[axis]+.001}))-base)/.001
                               for axis in ('pan','tilt')])
        if not np.isfinite(jac).all() or np.linalg.cond(jac) > 1e6:
            raise ValueError('PTZ aiming near an angular singularity')
        delta = -np.linalg.solve(jac,error)*gain
        return float(delta[0]),float(delta[1])


def matched_pixels(images, boxes, ratio=.7,foreground_masks=None):
    """Mutual local SIFT features, never independent bounding-box centres."""
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=.025)
    masks = [None,None] if foreground_masks is None else list(foreground_masks)
    if len(images)!=2 or len(boxes)!=2 or len(masks)!=2:
        raise ValueError('Dynamic matching requires exactly two images, boxes and masks')
    keypoints, descriptors = [], []
    for image, box,foreground in zip(images, boxes,masks):
        mask = np.zeros(image.shape[:2], np.uint8)
        x1, y1, x2, y2 = box
        dx, dy = .15*(x2-x1), .1*(y2-y1)
        xa, xb = max(0, int(x1+dx)), min(image.shape[1], int(x2-dx))
        ya, yb = max(0, int(y1+dy)), min(image.shape[0], int(y2-dy))
        if xa < xb and ya < yb:
            mask[ya:yb, xa:xb] = 255
        if foreground is not None:
            if np.asarray(foreground).shape != mask.shape:
                raise ValueError('Foreground mask dimensions differ from camera image')
            mask[np.asarray(foreground)==0] = 0
        kp, desc = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mask)
        keypoints.append(kp)
        descriptors.append(desc)
    if any(desc is None or len(desc) < 2 for desc in descriptors):
        return np.empty((0, 2)), np.empty((0, 2))
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def match(a, b):
        return {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(a, b, k=2)
                if len(pair) == 2 for m, n in [pair] if m.distance < ratio*n.distance}
    forward, reverse = match(*descriptors), match(*descriptors[::-1])
    pairs = [(i, j) for i, j in forward.items() if reverse.get(j) == i]
    return tuple(np.array([keypoints[side][pair[side]].pt for pair in pairs], float).reshape(-1, 2)
                 for side in (0, 1))


def surface_measurement(geometry, images, boxes, positions, limits, timing,foreground_masks=None):
    if any((image.shape[1], image.shape[0]) != geometry.size for image in images):
        raise ValueError('Dynamic image dimensions changed')
    a, b = matched_pixels(images, boxes, limits['ratio_test'],foreground_masks)
    xyz, valid, error, angle = geometry.triangulate(a, b, positions, limits)
    xyz, a, b, error, angle = xyz[valid], a[valid], b[valid], error[valid], angle[valid]
    minimum = limits['min_feature_points']
    if len(xyz) < minimum:
        return None, 'insufficient_common_surface_features'
    centres = [geometry.camera_pose(i, p)[1] for i, p in enumerate(positions)]
    ranges = np.linalg.norm(xyz-centres[0], axis=1)
    q10, q90 = np.percentile(ranges, [10, 90])
    if (q90-q10)/np.median(ranges) > limits['max_relative_depth_spread']:
        return None, 'mixed_surface_depths'
    med = np.median(ranges)
    mad = np.median(abs(ranges-med))
    keep = abs(ranges-med) <= max(3*1.4826*mad, .005*med)
    xyz, a, b, angle = xyz[keep], a[keep], b[keep], angle[keep]
    if len(xyz) < minimum or any(np.ptp(p[:, 0]) < 8 or np.ptp(p[:, 1]) < 8 for p in (a, b)):
        return None, 'insufficient_spatial_support'
    point = np.median(xyz, axis=0)
    lr, rr = [float(np.linalg.norm(point-c)) for c in centres]
    parallax = float(np.median(angle))
    focal = min(min(np.asarray(c['K'])[0, 0], np.asarray(c['K'])[1, 1]) for c in geometry.cameras)
    angular = math.radians(sum(c['angle_error_deg'] for c in geometry.cameras) + timing['pose_error_deg'])
    angular += 2*limits['disparity_sigma_px']/focal
    amplification = 1/max(math.sin(math.radians(parallax)), 1e-9)
    sigma = (max(lr, rr)*angular + timing['motion_error_m'] +
             sum(c['centre_error_m'] for c in geometry.cameras))*amplification
    return dict(xyz_world_m=point.tolist(), range_left_m=lr, range_right_m=rr,
                support=len(xyz), parallax_deg=parallax, depth_spread_m=float(q90-q10),
                approximate_error_budget_m=float(sigma), within_20cm_budget=bool(sigma <= .20),
                quality='estimated_with_empirical_timing_and_pose_model',
                support_region='foreground_masks' if foreground_masks is not None and all(m is not None for m in foreground_masks) else 'box_interiors',
                calibration_hash=getattr(geometry, 'file_hash', None)), 'estimated'
