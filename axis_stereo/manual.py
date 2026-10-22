"""Click the SAME stationary physical point in both RECTIFIED images."""
import json
from datetime import datetime,timezone
import cv2
import numpy as np
from .config import require_geometry,local
from .geometry import StereoGeometry
from .stream import StereoSource,GeometryGuard


def measure(cfg):
    require_geometry(cfg,static=True)
    geo = StereoGeometry.load(cfg)
    guard = GeometryGuard(cfg,geo.meta['ptz_reference'])
    try:
        guard.check(force=True)
        with StereoSource(cfg) as source:
            a,b = source.pair()
            guard.check(force=True)
        images = geo.rectify(a.image,b.image)
        points = [None,None]
        def click(side):
            def callback(event,x,y,flags,param):
                if event==cv2.EVENT_LBUTTONDOWN:
                    points[side]=(x,y)
            return callback
        for i in (0,1):
            title = ['Left point','Right SAME point'][i]
            cv2.namedWindow(title)
            cv2.setMouseCallback(title,click(i))
        print('Frozen static pair. Click SAME physical point in each view; ENTER calculates; Q quits.')
        while True:
            for i in (0,1):
                im = images[i].copy()
                if points[i] is not None:
                    cv2.drawMarker(im,points[i],(0,255,0),cv2.MARKER_CROSS,20,2)
                cv2.imshow(['Left point','Right SAME point'][i],im)
            key = cv2.waitKey(20)&255
            if key==ord('q'):
                break
            if key in (10,13) and all(p is not None for p in points):
                xyz,valid,error = geo.triangulate([points[0]],[points[1]],cfg['measurement'])
                if not valid[0]:
                    print('Rejected: invalid disparity, epipolar error, parallax or range')
                    continue
                l,r = geo.ranges(xyz[0])
                item = dict(utc=datetime.now(timezone.utc).isoformat(),left_px=points[0],right_px=points[1],
                            xyz_left_m=xyz[0].tolist(),range_left_m=float(l),range_right_m=float(r),
                            reprojection_error_px=float(error[0]))
                print(json.dumps(item,indent=2))
                dest = local(cfg,cfg['output']['directory'])
                dest.mkdir(parents=True,exist_ok=True)
                with (dest/'manual_measurements.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(item)+'\n')
    finally:
        guard.close()
        cv2.destroyAllWindows()
