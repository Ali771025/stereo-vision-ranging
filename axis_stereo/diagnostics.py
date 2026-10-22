import json
import platform
import time
import cv2
from .axis import AxisHTTP,open_rtsp,port_open,redacted_error
from .config import local
from .stream import StereoSource


def doctor(cfg,channels):
    dest = local(cfg,cfg['output']['directory'])/'diagnostics'
    dest.mkdir(parents=True,exist_ok=True)
    build = cv2.getBuildInformation()
    result = {'python':platform.python_version(),'opencv':cv2.__version__,
              'ffmpeg_build':[s.strip() for s in build.splitlines() if 'FFMPEG:' in s], 'cameras':{}}
    for side in ('left','right'):
        camera = cfg['cameras'][side]
        report = {'host':camera['host'],'ports':{},'channels':{}}
        for label,key in [('rtsp','rtsp_port'),('http','http_port')]:
            report['ports'][label] = port_open(camera['host'],camera[key])
        for channel in channels:
            c = dict(camera,channel=channel)
            item = {}
            client = None
            try:
                client = AxisHTTP(c)
                item['ptz'] = client.position()
                frame = client.snapshot(cfg['stream'])
                item['http_snapshot'] = 'OK'
                item['snapshot_size'] = [frame.shape[1],frame.shape[0]]
                cv2.imwrite(str(dest/f'{side}_ch{channel}_snapshot.jpg'),frame)
            except Exception as exc:
                item['http_snapshot'] = redacted_error(exc)
            finally:
                if client:
                    client.close()
            cap = None
            try:
                cap = open_rtsp(c,cfg['stream'])
                good = 0
                start = time.monotonic()
                while cap.isOpened() and good<5 and time.monotonic()-start<10:
                    ok,frame = cap.read()
                    if not ok:
                        break
                    good += 1
                item['rtsp_decoded_frames'] = good
                if good:
                    item['rtsp_size'] = [frame.shape[1],frame.shape[0]]
                    cv2.imwrite(str(dest/f'{side}_ch{channel}_rtsp.jpg'),frame)
            except Exception as exc:
                item['rtsp_error'] = redacted_error(exc)
            finally:
                if cap is not None:
                    cap.release()
            report['channels'][str(channel)] = item
        result['cameras'][side] = report
    try:
        import torch
        result['torch'] = str(torch.__version__)
        result['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result['gpu'] = torch.cuda.get_device_name(0)
    except ImportError:
        result['torch'] = 'not installed'
    (dest/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    print('Saved diagnostics to',dest)


def preview(cfg):
    with StereoSource(cfg) as source:
        try:
            while True:
                try:
                    a,b = source.pair()
                except TimeoutError as exc:
                    print(str(exc))
                    if cv2.waitKey(10)&255==ord('q'):
                        break
                    continue
                cv2.imshow('LEFT 192.0.2.10',a.image)
                cv2.imshow('RIGHT 192.0.2.10',b.image)
                if cv2.waitKey(1)&255==ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()
