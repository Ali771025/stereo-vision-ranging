from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Detection:
    box: tuple
    cls: int
    confidence: float
    name: str
    mask: np.ndarray | None = None


def inside(points, box, margin=0.1):
    x1,y1,x2,y2 = box
    dx,dy = (x2-x1)*margin,(y2-y1)*margin
    return ((points[:,0]>x1+dx)&(points[:,0]<x2-dx)&
            (points[:,1]>y1+dy)&(points[:,1]<y2-dy))


def supported(points,detection):
    result = inside(points,detection.box)
    if detection.mask is not None:
        mask = np.asarray(detection.mask)
        if mask.ndim != 2:
            raise ValueError('Foreground mask must be a 2D image')
        xy = np.rint(points).astype(int)
        valid = (xy[:,0]>=0)&(xy[:,0]<mask.shape[1])&(xy[:,1]>=0)&(xy[:,1]<mask.shape[0])
        foreground = np.zeros(len(points),bool)
        foreground[valid] = mask[xy[valid,1],xy[valid,0]]>0
        result &= foreground
    return result


def features(left,right,geometry,limits):
    sift = cv2.SIFT_create(nfeatures=5000,contrastThreshold=0.025)
    kp,desc = [],[]
    for im in (left,right):
        k,d = sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None)
        kp.append(k)
        desc.append(d)
    if any(d is None or len(d)<2 for d in desc):
        return np.empty((0,2)),np.empty((0,2)),np.empty((0,3))
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def ratio_matches(a,b):
        return {m.queryIdx:m.trainIdx for pair in matcher.knnMatch(a,b,k=2)
                if len(pair)==2 for m,n in [pair] if m.distance < limits['ratio_test']*n.distance}
    forward,backward = ratio_matches(*desc),ratio_matches(*desc[::-1])
    pairs = [(i,j) for i,j in forward.items() if backward.get(j)==i]
    a = np.array([kp[0][i].pt for i,j in pairs],np.float64).reshape(-1,2)
    b = np.array([kp[1][j].pt for i,j in pairs],np.float64).reshape(-1,2)
    xyz,valid,_ = geometry.triangulate(a,b,limits)
    return a[valid],b[valid],xyz[valid]


def associate(left_dets,right_dets,a,b,xyz,geometry,limits):
    """Conservative mutual best box association, supported by common physical features."""
    candidates = []
    for i,dl in enumerate(left_dets):
        for j,dr in enumerate(right_dets):
            if dl.cls != dr.cls:
                continue
            mask = supported(a,dl)&supported(b,dr)
            pts = xyz[mask]
            if len(pts) < limits['min_feature_points']:
                continue
            # Depth consistency is tested BEFORE trimming. A mixed foreground/background
            # box must not acquire a plausible-looking small background cluster.
            rect = pts@geometry.R1.T
            z = rect[:,2]
            q10,q90 = np.percentile(z,[10,90])
            if (q90-q10)/np.median(z) > limits['max_relative_depth_spread']:
                continue
            med = np.median(z)
            mad = np.median(abs(z-med))
            keep = abs(z-med)<=max(3*1.4826*mad,0.005*med)
            pts = pts[keep]
            if len(pts)<limits['min_feature_points']:
                continue
            local_a = a[mask][keep]
            local_b = b[mask][keep]
            # Require spatial support, not duplicate keypoints at one tiny spot.
            if np.ptp(local_a[:,0]) < 8 or np.ptp(local_a[:,1]) < 8:
                continue
            centre = np.median(pts,axis=0)
            lr,rr = geometry.ranges(centre)
            disparity = float(np.median(abs(local_a[:,0]-local_b[:,0])))
            candidates.append(dict(left_index=i,right_index=j,support=len(pts),
                xyz_left_m=centre.tolist(),range_left_m=float(lr),range_right_m=float(rr),
                depth_left_m=float(centre[2]),depth_rectified_m=float((geometry.R1@centre)[2]),
                disparity_px=disparity,
                approx_depth_sigma_m=float(med*limits['disparity_sigma_px']/disparity),
                depth_spread_m=float(q90-q10),
                feature_left_px=np.median(local_a,axis=0).tolist(),
                feature_right_px=np.median(local_b,axis=0).tolist()))
    results = []
    ratio = limits['ambiguity_ratio']
    for c in candidates:
        rivals = [r for r in candidates if r is not c and
                  (r['left_index']==c['left_index'] or r['right_index']==c['right_index'])]
        if rivals and c['support'] < ratio*max(r['support'] for r in rivals):
            continue
        c['class_id'] = left_dets[c['left_index']].cls
        c['class_name'] = left_dets[c['left_index']].name
        results.append(c)
    return results
