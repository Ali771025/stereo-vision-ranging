import csv
import json
import time
from datetime import datetime,timezone
import cv2
import numpy as np
from .config import local,require_geometry
from .geometry import StereoGeometry
from .matching import Detection,features,associate
from .stream import StereoSource,GeometryGuard
from .compute import prepare_device


class Detector:
    def __init__(self,cfg):
        path = local(cfg,cfg['model']['weights'])
        if not path.is_file():
            raise ValueError(f'Place your downloaded YOLO .pt weights at {path}; automatic download disabled')
        self.device, self.device_report = prepare_device(cfg['model']['device'])
        print('Inference device:',json.dumps(self.device_report))
        from ultralytics import YOLO
        self.model = YOLO(str(path),task=cfg['model'].get('task','detect'))
        # Pass torch.device directly so Ultralytics need not remap GPU indices
        # through CUDA_VISIBLE_DEVICES after CUDA has already been initialised.
        self.model.to(self.device)
        self.cfg = cfg['model']

    def predict(self,images):
        predictions = self.model.predict(images,device=self.device,imgsz=self.cfg['imgsz'],
             conf=self.cfg['confidence'],classes=self.cfg['classes'],verbose=False)
        results = []
        if len(predictions) != len(images):
            raise ValueError('Detector did not return one prediction per camera image')
        for pred,image in zip(predictions,images):
            detections = []
            if pred.boxes is None:
                raise ValueError('Model must provide detection boxes; use a local detection or segmentation checkpoint')
            polygons = getattr(getattr(pred,'masks',None),'xy',None)
            for index,box in enumerate(pred.boxes):
                cls = int(box.cls.item())
                mask = None
                if polygons is not None and index < len(polygons):
                    mask = np.zeros(image.shape[:2],np.uint8)
                    polygon = np.asarray(polygons[index],np.float64).reshape(-1,2)
                    if len(polygon)>=3 and np.isfinite(polygon).all():
                        cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],255)
                detections.append(Detection(tuple(box.xyxy[0].cpu().tolist()),cls,
                                             float(box.conf.item()),pred.names[cls],mask))
            results.append(detections)
        return results


def annotate(images,dets,measurements,status):
    views = [im.copy() for im in images]
    for side,(im,ds) in enumerate(zip(views,dets)):
        index_key = 'left_index' if side==0 else 'right_index'
        range_key = 'range_left_m' if side==0 else 'range_right_m'
        for i,d in enumerate(ds):
            x1,y1,x2,y2 = map(int,d.box)
            ms = [m for m in measurements if m[index_key]==i]
            color = (0,220,0) if ms else (0,180,255)
            label = f'{d.name} {d.confidence:.2f} | ' + (f'{ms[0][range_key]:.2f}m' if ms else 'range N/A')
            cv2.rectangle(im,(x1,y1),(x2,y2),color,2)
            cv2.putText(im,label,(max(0,x1),max(20,y1-7)),0,0.55,color,2)
        cv2.putText(im,status,(10,25),0,0.6,(0,255,255),2)
    return np.hstack(views)


def run(cfg,detect_only=False,headless=False,max_frames=None):
    if max_frames is not None and max_frames <= 0:
        raise ValueError('max_frames must be positive')
    geometry = None
    guard = None
    writer = None
    count = 0
    fields = ['utc','frame','class_name','range_left_m','range_right_m','depth_left_m',
              'depth_rectified_m','support','disparity_px','approx_depth_sigma_m','arrival_skew_ms']
    try:
        if not detect_only:
            require_geometry(cfg,static=True)
            geometry = StereoGeometry.load(cfg)
            guard = GeometryGuard(cfg,geometry.meta['ptz_reference'])
            guard.check(force=True)
        detector = Detector(cfg)
        out = local(cfg,cfg['output']['directory'])
        out.mkdir(parents=True,exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        with (out/f'{stamp}.jsonl').open('w',encoding='utf-8') as log, \
             (out/f'{stamp}.csv').open('w',newline='',encoding='utf-8-sig') as cf, StereoSource(cfg) as source:
            csv_writer = csv.DictWriter(cf,fieldnames=fields)
            csv_writer.writeheader()
            while max_frames is None or count<max_frames:
                try:
                    a,b = source.pair()
                except TimeoutError as exc:
                    print(str(exc))
                    log.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
                                              event='stream_wait',measurements=[]))+'\n')
                    log.flush()
                    if not headless:
                        width,height = geometry.size if geometry else tuple(map(int,cfg['stream']['resolution'].split('x')))
                        unavailable = np.zeros((height,width*2,3),np.uint8)
                        cv2.putText(unavailable,'RECONNECTING | no current measurements',(15,40),0,0.8,(0,180,255),2)
                        cv2.imshow('Axis stereo YOLO | Q quit | S snapshot',unavailable)
                        if cv2.waitKey(10)&255==ord('q'):
                            break
                    continue
                if guard:
                    guard.check()
                images = (a.image,b.image)
                if images[0].shape != images[1].shape:
                    raise ValueError('Camera dimensions differ')
                if geometry:
                    images = geometry.rectify(*images)
                dets = detector.predict(list(images))
                measurements = []
                status = 'DETECTION ONLY'
                if geometry:
                    points = features(*images,geometry,cfg['measurement'])
                    measurements = associate(*dets,*points,geometry,cfg['measurement'])
                    status = 'STATIC SCENE | receive-time pairing'
                if guard:
                    guard.check(force=True)
                # Inference can be slow. Do not publish stale numeric measurements.
                fresh = time.monotonic()-min(a.arrival,b.arrival) <= cfg['stream']['max_age_ms']/1000
                if not fresh:
                    measurements = []
                    status = 'STALE processing result: range N/A'
                count += 1
                utc = datetime.now(timezone.utc).isoformat()
                skew = abs(a.arrival-b.arrival)*1000
                record = dict(utc=utc,frame=count,arrival_skew_ms=skew,
                              pairing='host_receive_time_not_exposure_sync',fresh=fresh,
                              detections=[[dict(box=d.box,cls=d.cls,name=d.name,confidence=d.confidence)
                                           for d in side] for side in dets],measurements=measurements)
                log.write(json.dumps(record,ensure_ascii=False)+'\n')
                log.flush()
                for m in measurements:
                    csv_writer.writerow({k: (dict(utc=utc,frame=count,arrival_skew_ms=skew)|m).get(k) for k in fields})
                cf.flush()
                canvas = annotate(images,dets,measurements,status)
                if cfg['output']['save_video']:
                    if writer is None:
                        writer = cv2.VideoWriter(str(out/f'{stamp}.avi'),cv2.VideoWriter_fourcc(*'MJPG'),
                                                 cfg['stream']['fps'],(canvas.shape[1],canvas.shape[0]))
                        if not writer.isOpened():
                            raise OSError('Could not create video output')
                    writer.write(canvas)
                if not headless:
                    cv2.imshow('Axis stereo YOLO | Q quit | S snapshot',canvas)
                    key = cv2.waitKey(1)&255
                    if key==ord('q'):
                        break
                    if key==ord('s'):
                        cv2.imwrite(str(out/f'{stamp}_{count}.jpg'),canvas)
                if count%20==0:
                    print(f'frame={count} measurements={len(measurements)} receive_skew={skew:.1f}ms')
    finally:
        if guard:
            guard.close()
        if writer:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
