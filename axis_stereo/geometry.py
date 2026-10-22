import json
import cv2
import numpy as np
from .config import fingerprint, local


class StereoGeometry:
    def __init__(self, data):
        self.data = data
        for k in ('K1','D1','K2','D2','R','T','R1','R2','P1','P2'):
            setattr(self, k, np.asarray(data[k], dtype=np.float64))
            if not np.isfinite(getattr(self,k)).all():
                raise ValueError('Nonfinite calibration parameter: ' + k)
        for k,shape in dict(K1=(3,3),K2=(3,3),R=(3,3),R1=(3,3),R2=(3,3),P1=(3,4),P2=(3,4)).items():
            if getattr(self,k).shape != shape:
                raise ValueError('Wrong calibration matrix dimensions: ' + k)
        self.T = self.T.reshape(3)
        self.size = tuple(int(x) for x in data['size'])
        if len(self.size) != 2 or min(self.size) <= 0:
            raise ValueError('Invalid calibration image dimensions')
        self.maps = [cv2.initUndistortRectifyMap(K, D, R, P, self.size, cv2.CV_32FC1)
                     for K,D,R,P in [(self.K1,self.D1,self.R1,self.P1),(self.K2,self.D2,self.R2,self.P2)]]
        # Only horizontal rigs are supported by feature correspondence gates.
        if abs(self.P2[1,3]) > abs(self.P2[0,3]) or abs(self.P2[0,3]) < 1e-9:
            raise ValueError('Requires nonzero predominantly horizontal stereo baseline')
        self.baseline = float(np.linalg.norm(self.T))
        if not np.isfinite(self.baseline) or self.baseline <= 0:
            raise ValueError('Invalid baseline')

    @classmethod
    def load(cls, cfg):
        with np.load(local(cfg, cfg['calibration']['file']), allow_pickle=False) as f:
            d = {k: f[k] for k in f.files}
        meta = json.loads(str(d['metadata']))
        if meta['fingerprint'] != fingerprint(cfg):
            raise ValueError('Calibration is for different camera/stream/rig settings')
        if not meta.get('validated'):
            raise ValueError('Calibration not validated')
        obj = cls(d)
        obj.meta = meta
        return obj

    def rectify(self, left, right):
        if any((im.shape[1],im.shape[0]) != self.size for im in (left,right)):
            raise ValueError('Actual image dimensions changed; recalibration required')
        return tuple(cv2.remap(im,*maps,cv2.INTER_LINEAR) for im,maps in zip((left,right),self.maps))

    def triangulate(self, a, b, limits):
        a, b = np.asarray(a, np.float64).reshape(-1,2), np.asarray(b, np.float64).reshape(-1,2)
        if len(a) != len(b):
            raise ValueError('Correspondence count mismatch')
        if not len(a):
            return np.empty((0,3)), np.zeros(0, bool), np.zeros(0)
        h = cv2.triangulatePoints(self.P1,self.P2,a.T,b.T).T
        valid = np.abs(h[:,3]) > 1e-10
        xyz = np.full((len(a),3),np.nan)
        xyz[valid] = h[valid,:3]/h[valid,3,None]
        orig = xyz @ self.R1  # row-vector equivalent of R1.T @ X_rect
        right = orig @ self.R.T + self.T
        z = xyz[:,2]
        d = a[:,0]-b[:,0]
        expected_sign = np.sign(-self.P2[0,3])
        valid &= np.isfinite(xyz).all(axis=1)
        valid &= (orig[:,2]>0)&(right[:,2]>0)
        valid &= (z>=limits['min_depth_m'])&(z<=limits['max_depth_m'])
        valid &= (d*expected_sign>=limits['min_disparity_px'])&(np.abs(d)<=limits['max_disparity_px'])
        valid &= np.abs(a[:,1]-b[:,1])<=limits['max_epipolar_px']
        error = np.full(len(a),np.inf)
        xh = np.column_stack([xyz,np.ones(len(a))])
        with np.errstate(divide='ignore',invalid='ignore'):
            lproj, rproj = xh@self.P1.T, xh@self.P2.T
            errors = [np.linalg.norm(p[:,:2]/p[:,2,None]-obs,axis=1) for p,obs in ((lproj,a),(rproj,b))]
            error = np.maximum(*errors)
            c_right_rect = -np.linalg.inv(self.P2[:,:3])@self.P2[:,3]
            ray1,ray2 = xyz, xyz-c_right_rect
            cosine = np.sum(ray1*ray2,axis=1)/(np.linalg.norm(ray1,axis=1)*np.linalg.norm(ray2,axis=1))
            angle = np.degrees(np.arccos(np.clip(cosine,-1,1)))
        valid &= (error<=limits['max_reprojection_px'])&(angle>=limits['min_parallax_deg'])
        return orig, valid, error

    def ranges(self, xyz):
        xyz = np.asarray(xyz)
        return np.linalg.norm(xyz,axis=-1), np.linalg.norm(xyz@self.R.T+self.T,axis=-1)
